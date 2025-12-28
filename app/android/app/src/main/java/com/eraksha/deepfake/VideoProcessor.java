package com.eraksha.deepfake;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.util.Log;

import java.io.IOException;

public class VideoProcessor {
    private static final String TAG = "VideoProcessor";
    private Context context;
    
    public VideoProcessor(Context context) {
        this.context = context;
    }
    
    /**
     * Extract frames from video
     * @param videoUri URI of the video
     * @param numFrames Number of frames to extract
     * @return 4D array [batch, frames, channels, height, width]
     */
    public float[][][][] extractFrames(Uri videoUri, int numFrames) throws IOException {
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        
        try {
            retriever.setDataSource(context, videoUri);
            
            // Get video duration
            String durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
            long duration = Long.parseLong(durationStr) * 1000; // Convert to microseconds
            
            // Calculate frame timestamps
            long[] timestamps = new long[numFrames];
            for (int i = 0; i < numFrames; i++) {
                timestamps[i] = (duration * i) / (numFrames - 1);
            }
            
            // Extract frames
            float[][][][] frames = new float[1][numFrames][3][224][224]; // [batch, frames, channels, height, width]
            
            for (int i = 0; i < numFrames; i++) {
                Bitmap frame = retriever.getFrameAtTime(timestamps[i], MediaMetadataRetriever.OPTION_CLOSEST_SYNC);
                
                if (frame != null) {
                    // Resize frame to 224x224
                    Bitmap resizedFrame = Bitmap.createScaledBitmap(frame, 224, 224, true);
                    
                    // Convert to float array and normalize
                    convertBitmapToFloatArray(resizedFrame, frames[0][i]);
                    
                    frame.recycle();
                    resizedFrame.recycle();
                } else {
                    Log.w(TAG, "Failed to extract frame at timestamp: " + timestamps[i]);
                    // Fill with zeros if frame extraction fails
                    for (int c = 0; c < 3; c++) {
                        for (int h = 0; h < 224; h++) {
                            for (int w = 0; w < 224; w++) {
                                frames[0][i][c][h][w] = 0.0f;
                            }
                        }
                    }
                }
            }
            
            Log.d(TAG, "Successfully extracted " + numFrames + " frames");
            return frames;
            
        } catch (Exception e) {
            Log.e(TAG, "Error extracting frames", e);
            throw new IOException("Failed to extract video frames", e);
        } finally {
            try {
                retriever.release();
            } catch (Exception e) {
                Log.w(TAG, "Error releasing MediaMetadataRetriever", e);
            }
        }
    }
    
    /**
     * Convert bitmap to float array in CHW format
     * @param bitmap Input bitmap
     * @param output Output array [channels][height][width]
     */
    private void convertBitmapToFloatArray(Bitmap bitmap, float[][][] output) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int pixel = pixels[h * width + w];
                
                // Extract RGB values and normalize to [0, 1]
                float r = ((pixel >> 16) & 0xFF) / 255.0f;
                float g = ((pixel >> 8) & 0xFF) / 255.0f;
                float b = (pixel & 0xFF) / 255.0f;
                
                // Store in CHW format
                output[0][h][w] = r; // Red channel
                output[1][h][w] = g; // Green channel
                output[2][h][w] = b; // Blue channel
            }
        }
    }
    
    /**
     * Get video metadata
     * @param videoUri URI of the video
     * @return VideoMetadata object
     */
    public VideoMetadata getVideoMetadata(Uri videoUri) {
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        
        try {
            retriever.setDataSource(context, videoUri);
            
            String durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
            String widthStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH);
            String heightStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT);
            String frameRateStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE);
            
            long duration = durationStr != null ? Long.parseLong(durationStr) : 0;
            int width = widthStr != null ? Integer.parseInt(widthStr) : 0;
            int height = heightStr != null ? Integer.parseInt(heightStr) : 0;
            float frameRate = frameRateStr != null ? Float.parseFloat(frameRateStr) : 0;
            
            return new VideoMetadata(duration, width, height, frameRate);
            
        } catch (Exception e) {
            Log.e(TAG, "Error getting video metadata", e);
            return new VideoMetadata(0, 0, 0, 0);
        } finally {
            try {
                retriever.release();
            } catch (Exception e) {
                Log.w(TAG, "Error releasing MediaMetadataRetriever", e);
            }
        }
    }
    
    /**
     * Video metadata container
     */
    public static class VideoMetadata {
        public final long duration; // milliseconds
        public final int width;
        public final int height;
        public final float frameRate;
        
        public VideoMetadata(long duration, int width, int height, float frameRate) {
            this.duration = duration;
            this.width = width;
            this.height = height;
            this.frameRate = frameRate;
        }
        
        @Override
        public String toString() {
            return String.format("VideoMetadata{duration=%dms, size=%dx%d, fps=%.1f}", 
                               duration, width, height, frameRate);
        }
    }
}