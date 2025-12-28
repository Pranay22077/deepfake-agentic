package com.eraksha.deepfake;

import android.content.Context;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.util.Log;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;

public class AudioProcessor {
    private static final String TAG = "AudioProcessor";
    private static final int TARGET_SAMPLE_RATE = 16000;
    
    private Context context;
    
    public AudioProcessor(Context context) {
        this.context = context;
    }
    
    /**
     * Extract audio waveform from video
     * @param videoUri URI of the video
     * @param durationSeconds Duration of audio to extract
     * @return Audio waveform as float array
     */
    public float[] extractAudio(Uri videoUri, float durationSeconds) throws IOException {
        MediaExtractor extractor = new MediaExtractor();
        
        try {
            extractor.setDataSource(context, videoUri, null);
            
            // Find audio track
            int audioTrackIndex = -1;
            MediaFormat audioFormat = null;
            
            for (int i = 0; i < extractor.getTrackCount(); i++) {
                MediaFormat format = extractor.getTrackFormat(i);
                String mime = format.getString(MediaFormat.KEY_MIME);
                
                if (mime != null && mime.startsWith("audio/")) {
                    audioTrackIndex = i;
                    audioFormat = format;
                    break;
                }
            }
            
            if (audioTrackIndex == -1) {
                Log.w(TAG, "No audio track found, returning silence");
                return createSilence(durationSeconds);
            }
            
            // Select audio track
            extractor.selectTrack(audioTrackIndex);
            
            // Get audio properties
            int sampleRate = audioFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE);
            int channelCount = audioFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
            
            Log.d(TAG, String.format("Audio format: %d Hz, %d channels", sampleRate, channelCount));
            
            // Calculate target sample count
            int targetSamples = (int) (TARGET_SAMPLE_RATE * durationSeconds);
            float[] audioData = new float[targetSamples];
            
            // Extract audio data
            ByteBuffer buffer = ByteBuffer.allocate(1024 * 1024); // 1MB buffer
            int sampleIndex = 0;
            
            while (sampleIndex < targetSamples) {
                buffer.clear();
                int sampleSize = extractor.readSampleData(buffer, 0);
                
                if (sampleSize < 0) {
                    // End of stream
                    break;
                }
                
                // Convert bytes to audio samples
                buffer.flip();
                buffer.order(ByteOrder.LITTLE_ENDIAN);
                
                // Assume 16-bit PCM audio
                ShortBuffer shortBuffer = buffer.asShortBuffer();
                
                while (shortBuffer.hasRemaining() && sampleIndex < targetSamples) {
                    short sample = shortBuffer.get();
                    
                    // Convert to float and normalize
                    float floatSample = sample / 32768.0f;
                    
                    // Handle multi-channel audio (convert to mono)
                    if (channelCount > 1) {
                        // Average channels for mono conversion
                        float sum = floatSample;
                        for (int c = 1; c < channelCount && shortBuffer.hasRemaining(); c++) {
                            sum += shortBuffer.get() / 32768.0f;
                        }
                        floatSample = sum / channelCount;
                    }
                    
                    audioData[sampleIndex++] = floatSample;
                }
                
                extractor.advance();
            }
            
            // Resample if necessary
            if (sampleRate != TARGET_SAMPLE_RATE) {
                audioData = resampleAudio(audioData, sampleIndex, sampleRate, TARGET_SAMPLE_RATE);
            }
            
            // Pad with zeros if we didn't get enough samples
            if (sampleIndex < targetSamples) {
                for (int i = sampleIndex; i < targetSamples; i++) {
                    audioData[i] = 0.0f;
                }
            }
            
            Log.d(TAG, String.format("Extracted %d audio samples", targetSamples));
            return audioData;
            
        } catch (Exception e) {
            Log.e(TAG, "Error extracting audio", e);
            throw new IOException("Failed to extract audio", e);
        } finally {
            try {
                extractor.release();
            } catch (Exception e) {
                Log.w(TAG, "Error releasing MediaExtractor", e);
            }
        }
    }
    
    /**
     * Create silence audio
     * @param durationSeconds Duration in seconds
     * @return Silent audio waveform
     */
    private float[] createSilence(float durationSeconds) {
        int samples = (int) (TARGET_SAMPLE_RATE * durationSeconds);
        float[] silence = new float[samples];
        // Array is already initialized to zeros
        Log.d(TAG, "Created silence audio with " + samples + " samples");
        return silence;
    }
    
    /**
     * Simple linear resampling
     * @param input Input audio data
     * @param inputLength Actual length of input data
     * @param inputSampleRate Input sample rate
     * @param outputSampleRate Output sample rate
     * @return Resampled audio data
     */
    private float[] resampleAudio(float[] input, int inputLength, int inputSampleRate, int outputSampleRate) {
        if (inputSampleRate == outputSampleRate) {
            return input;
        }
        
        double ratio = (double) inputSampleRate / outputSampleRate;
        int outputLength = (int) (inputLength / ratio);
        float[] output = new float[outputLength];
        
        for (int i = 0; i < outputLength; i++) {
            double sourceIndex = i * ratio;
            int index1 = (int) sourceIndex;
            int index2 = Math.min(index1 + 1, inputLength - 1);
            double fraction = sourceIndex - index1;
            
            // Linear interpolation
            output[i] = (float) (input[index1] * (1 - fraction) + input[index2] * fraction);
        }
        
        Log.d(TAG, String.format("Resampled audio from %d Hz to %d Hz (%d -> %d samples)", 
                                inputSampleRate, outputSampleRate, inputLength, outputLength));
        
        return output;
    }
    
    /**
     * Get audio metadata
     * @param videoUri URI of the video
     * @return AudioMetadata object
     */
    public AudioMetadata getAudioMetadata(Uri videoUri) {
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        
        try {
            retriever.setDataSource(context, videoUri);
            
            String durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
            String bitrateStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_BITRATE);
            
            long duration = durationStr != null ? Long.parseLong(durationStr) : 0;
            int bitrate = bitrateStr != null ? Integer.parseInt(bitrateStr) : 0;
            
            // Try to get sample rate from extractor
            int sampleRate = 0;
            int channelCount = 0;
            
            MediaExtractor extractor = new MediaExtractor();
            try {
                extractor.setDataSource(context, videoUri, null);
                
                for (int i = 0; i < extractor.getTrackCount(); i++) {
                    MediaFormat format = extractor.getTrackFormat(i);
                    String mime = format.getString(MediaFormat.KEY_MIME);
                    
                    if (mime != null && mime.startsWith("audio/")) {
                        sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE);
                        channelCount = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
                        break;
                    }
                }
            } finally {
                extractor.release();
            }
            
            return new AudioMetadata(duration, sampleRate, channelCount, bitrate);
            
        } catch (Exception e) {
            Log.e(TAG, "Error getting audio metadata", e);
            return new AudioMetadata(0, 0, 0, 0);
        } finally {
            try {
                retriever.release();
            } catch (Exception e) {
                Log.w(TAG, "Error releasing MediaMetadataRetriever", e);
            }
        }
    }
    
    /**
     * Audio metadata container
     */
    public static class AudioMetadata {
        public final long duration; // milliseconds
        public final int sampleRate;
        public final int channelCount;
        public final int bitrate;
        
        public AudioMetadata(long duration, int sampleRate, int channelCount, int bitrate) {
            this.duration = duration;
            this.sampleRate = sampleRate;
            this.channelCount = channelCount;
            this.bitrate = bitrate;
        }
        
        @Override
        public String toString() {
            return String.format("AudioMetadata{duration=%dms, rate=%dHz, channels=%d, bitrate=%d}", 
                               duration, sampleRate, channelCount, bitrate);
        }
    }
}