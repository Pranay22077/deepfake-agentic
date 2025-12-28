package com.eraksha.deepfake;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "E-Raksha";
    private static final int REQUEST_VIDEO_CAPTURE = 1;
    private static final int REQUEST_VIDEO_PICK = 2;
    private static final int PERMISSION_REQUEST_CODE = 100;
    
    private Module deepfakeModel;
    private Button btnRecord, btnSelect, btnAnalyze;
    private TextView tvResult, tvConfidence;
    private ProgressBar progressBar;
    private Uri selectedVideoUri;
    
    private VideoProcessor videoProcessor;
    private AudioProcessor audioProcessor;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        initializeViews();
        requestPermissions();
        loadModel();
        
        btnRecord.setOnClickListener(v -> recordVideo());
        btnSelect.setOnClickListener(v -> selectVideo());
        btnAnalyze.setOnClickListener(v -> analyzeVideo());
    }
    
    private void initializeViews() {
        btnRecord = findViewById(R.id.btn_record);
        btnSelect = findViewById(R.id.btn_select);
        btnAnalyze = findViewById(R.id.btn_analyze);
        tvResult = findViewById(R.id.tv_result);
        tvConfidence = findViewById(R.id.tv_confidence);
        progressBar = findViewById(R.id.progress_bar);
        
        videoProcessor = new VideoProcessor(this);
        audioProcessor = new AudioProcessor(this);
    }
    
    private void requestPermissions() {
        String[] permissions = {
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        };
        
        boolean allGranted = true;
        for (String permission : permissions) {
            if (ContextCompat.checkSelfPermission(this, permission) 
                != PackageManager.PERMISSION_GRANTED) {
                allGranted = false;
                break;
            }
        }
        
        if (!allGranted) {
            ActivityCompat.requestPermissions(this, permissions, PERMISSION_REQUEST_CODE);
        }
    }
    
    private void loadModel() {
        new Thread(() -> {
            try {
                // Copy model from assets to internal storage
                String modelPath = copyAssetToInternalStorage("baseline_student_mobile.ptl");
                
                // Load PyTorch Lite model
                deepfakeModel = LiteModuleLoader.load(modelPath);
                
                runOnUiThread(() -> {
                    Toast.makeText(this, "Model loaded successfully", Toast.LENGTH_SHORT).show();
                    btnAnalyze.setEnabled(true);
                });
                
                Log.d(TAG, "Model loaded successfully");
                
            } catch (IOException e) {
                Log.e(TAG, "Error loading model", e);
                runOnUiThread(() -> {
                    Toast.makeText(this, "Error loading model: " + e.getMessage(), 
                                 Toast.LENGTH_LONG).show();
                });
            }
        }).start();
    }
    
    private String copyAssetToInternalStorage(String assetName) throws IOException {
        File file = new File(getFilesDir(), assetName);
        
        if (!file.exists()) {
            try (InputStream inputStream = getAssets().open(assetName);
                 FileOutputStream outputStream = new FileOutputStream(file)) {
                
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }
            }
        }
        
        return file.getAbsolutePath();
    }
    
    private void recordVideo() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) 
            != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show();
            return;
        }
        
        Intent intent = new Intent(MediaStore.ACTION_VIDEO_CAPTURE);
        intent.putExtra(MediaStore.EXTRA_DURATION_LIMIT, 10); // 10 seconds max
        intent.putExtra(MediaStore.EXTRA_VIDEO_QUALITY, 1); // High quality
        
        if (intent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(intent, REQUEST_VIDEO_CAPTURE);
        } else {
            Toast.makeText(this, "No camera app available", Toast.LENGTH_SHORT).show();
        }
    }
    
    private void selectVideo() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Video.Media.EXTERNAL_CONTENT_URI);
        intent.setType("video/*");
        
        if (intent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(intent, REQUEST_VIDEO_PICK);
        } else {
            Toast.makeText(this, "No file manager available", Toast.LENGTH_SHORT).show();
        }
    }
    
    private void analyzeVideo() {
        if (selectedVideoUri == null) {
            Toast.makeText(this, "Please select or record a video first", Toast.LENGTH_SHORT).show();
            return;
        }
        
        if (deepfakeModel == null) {
            Toast.makeText(this, "Model not loaded yet", Toast.LENGTH_SHORT).show();
            return;
        }
        
        // Show progress
        progressBar.setVisibility(View.VISIBLE);
        btnAnalyze.setEnabled(false);
        tvResult.setText("Analyzing...");
        tvConfidence.setText("");
        
        // Run analysis in background thread
        new Thread(() -> {
            try {
                // Extract video frames
                float[][][][] videoFrames = videoProcessor.extractFrames(selectedVideoUri, 8);
                
                // Extract audio
                float[] audioWaveform = audioProcessor.extractAudio(selectedVideoUri, 3.0f);
                
                // Convert to tensors
                Tensor videoTensor = Tensor.fromBlob(videoFrames, new long[]{1, 8, 3, 224, 224});
                Tensor audioTensor = Tensor.fromBlob(audioWaveform, new long[]{1, audioWaveform.length});
                
                // Run inference
                IValue[] inputs = {IValue.from(videoTensor), IValue.from(audioTensor)};
                Tensor output = deepfakeModel.forward(IValue.tuple(inputs)).toTensor();
                
                // Get probabilities
                float[] scores = output.getDataAsFloatArray();
                float realProb = scores[0];
                float fakeProb = scores[1];
                
                // Apply softmax
                float expReal = (float) Math.exp(realProb);
                float expFake = (float) Math.exp(fakeProb);
                float sum = expReal + expFake;
                
                float realProbability = expReal / sum;
                float fakeProbability = expFake / sum;
                
                // Update UI
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    btnAnalyze.setEnabled(true);
                    
                    String result = fakeProbability > 0.5 ? "DEEPFAKE DETECTED" : "AUTHENTIC";
                    String confidence = String.format("Confidence: %.1f%%", 
                                                    Math.max(realProbability, fakeProbability) * 100);
                    
                    tvResult.setText(result);
                    tvConfidence.setText(confidence);
                    
                    // Set result color
                    int color = fakeProbability > 0.5 ? 
                               getResources().getColor(android.R.color.holo_red_dark) :
                               getResources().getColor(android.R.color.holo_green_dark);
                    tvResult.setTextColor(color);
                    
                    Log.d(TAG, String.format("Analysis complete: Real=%.3f, Fake=%.3f", 
                                            realProbability, fakeProbability));
                });
                
            } catch (Exception e) {
                Log.e(TAG, "Error during analysis", e);
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    btnAnalyze.setEnabled(true);
                    tvResult.setText("Analysis failed");
                    tvConfidence.setText("Error: " + e.getMessage());
                    tvResult.setTextColor(getResources().getColor(android.R.color.holo_red_dark));
                });
            }
        }).start();
    }
    
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        
        if (resultCode == RESULT_OK) {
            if (requestCode == REQUEST_VIDEO_CAPTURE || requestCode == REQUEST_VIDEO_PICK) {
                selectedVideoUri = data.getData();
                
                if (selectedVideoUri != null) {
                    Toast.makeText(this, "Video selected. Ready to analyze.", Toast.LENGTH_SHORT).show();
                    btnAnalyze.setEnabled(deepfakeModel != null);
                } else {
                    Toast.makeText(this, "Failed to get video", Toast.LENGTH_SHORT).show();
                }
            }
        }
    }
    
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, 
                                         @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        
        if (requestCode == PERMISSION_REQUEST_CODE) {
            boolean allGranted = true;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }
            
            if (!allGranted) {
                Toast.makeText(this, "Permissions required for app to function", 
                             Toast.LENGTH_LONG).show();
            }
        }
    }
}