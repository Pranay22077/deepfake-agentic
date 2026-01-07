# 🚀 CONTINUE IN NEW SESSION - QUICK START GUIDE

## 📋 WHAT'S BEEN DONE

✅ **10 out of 15 scripts complete (67%)**
- **AV Model:** 100% complete (all 3 stages) ✅
- **CM Model:** 100% complete (all 3 stages) ✅
- **BG Model:** 67% complete (2 out of 3 stages) 🔄

## 🎯 WHAT'S LEFT TO DO

### Priority 1: Complete BG Stage 4 (15 minutes)
**File:** `train_bg_stage4_dfdc.py` (currently ~50 lines, needs ~750 more)

**Action:** Copy the entire content from `train_cm_stage4_dfdc.py` and:
1. Replace all "CM" with "BG"
2. Replace all "cm_" with "bg_"
3. Replace all "Compression" with "Background/Lighting"
4. Replace `CompressionModule` with `BackgroundLightingModule`
5. Update `specialist_features = 44 * 7 * 7  # 2156` (instead of 40 * 7 * 7)

### Priority 2: Create RR Model (3 scripts, ~45 minutes)

#### RR Stage 1: `train_rr_stage1_faceforensics.py`
**Template:** Copy `train_cm_stage1_faceforensics.py`

**Changes:**
1. Replace header: "COMPRESSION (CM)" → "RESOLUTION (RR)"
2. Replace all "cm_" → "rr_"
3. Replace all "CM" → "RR"
4. Replace `CompressionModule` with this:

```python
class ResolutionModule(nn.Module):
    """Resolution artifact detection module"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Multi-scale resolution analyzer
        self.resolution_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=3, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=1)
        )
        
        # Upscaling artifact detector
        self.upscaling_detector = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
        
        # Edge sharpness checker
        self.edge_checker = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
        
        # Pixel interpolation detector
        self.interpolation_detector = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=7, padding=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=7, padding=3),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=1)
        )
        
        # Attention fusion
        self.attention = nn.Sequential(
            nn.Conv2d(36, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 36, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        res_feat = F.adaptive_avg_pool2d(self.resolution_analyzer(x), (7, 7))
        upscale_feat = F.adaptive_avg_pool2d(self.upscaling_detector(x), (7, 7))
        edge_feat = F.adaptive_avg_pool2d(self.edge_checker(x), (7, 7))
        interp_feat = F.adaptive_avg_pool2d(self.interpolation_detector(x), (7, 7))
        combined = torch.cat([res_feat, upscale_feat, edge_feat, interp_feat], dim=1)
        return combined * self.attention(combined)
```

5. Update: `specialist_features = 36 * 7 * 7  # 1764`

#### RR Stage 2: `train_rr_stage2_celebdf.py`
**Template:** Copy `train_cm_stage2_celebdf.py`
**Changes:** Same as Stage 1 (replace CM → RR, update module, update features)

#### RR Stage 4: `train_rr_stage4_dfdc.py`
**Template:** Copy `train_cm_stage4_dfdc.py`
**Changes:** Same as Stage 1 (replace CM → RR, update module, update features)

### Priority 3: Create TM Model (3 scripts, ~45 minutes)

#### TM Stage 1: `train_tm_stage1_faceforensics.py`
**Template:** Copy `train_cm_stage1_faceforensics.py`

**Changes:**
1. Replace header: "COMPRESSION (CM)" → "TEMPORAL (TM)"
2. Replace all "cm_" → "tm_"
3. Replace all "CM" → "TM"
4. Replace `CompressionModule` with this:

```python
class TemporalModule(nn.Module):
    """Temporal inconsistency detection module"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Frame-to-frame consistency checker
        self.frame_consistency = nn.Sequential(
            nn.Conv2d(in_channels, 14, kernel_size=5, padding=2),
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.Conv2d(14, 28, kernel_size=5, padding=2),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.Conv2d(28, 14, kernel_size=1)
        )
        
        # Motion flow analyzer
        self.motion_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=7, padding=3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=7, padding=3),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Temporal artifact detector
        self.temporal_detector = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Optical flow validator
        self.optical_flow = nn.Sequential(
            nn.Conv2d(in_channels, 14, kernel_size=9, padding=4),
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.Conv2d(14, 28, kernel_size=9, padding=4),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.Conv2d(28, 14, kernel_size=1)
        )
        
        # Attention fusion
        self.attention = nn.Sequential(
            nn.Conv2d(52, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 52, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        frame_feat = F.adaptive_avg_pool2d(self.frame_consistency(x), (7, 7))
        motion_feat = F.adaptive_avg_pool2d(self.motion_analyzer(x), (7, 7))
        temporal_feat = F.adaptive_avg_pool2d(self.temporal_detector(x), (7, 7))
        optical_feat = F.adaptive_avg_pool2d(self.optical_flow(x), (7, 7))
        combined = torch.cat([frame_feat, motion_feat, temporal_feat, optical_feat], dim=1)
        return combined * self.attention(combined)
```

5. Update: `specialist_features = 52 * 7 * 7  # 2548`

#### TM Stage 2: `train_tm_stage2_celebdf.py`
**Template:** Copy `train_cm_stage2_celebdf.py`
**Changes:** Same as Stage 1 (replace CM → TM, update module, update features)

#### TM Stage 4: `train_tm_stage4_dfdc.py`
**Template:** Copy `train_cm_stage4_dfdc.py`
**Changes:** Same as Stage 1 (replace CM → TM, update module, update features)

---

## 📝 QUICK CHECKLIST

When creating each script:
- [ ] Copy the correct template file
- [ ] Replace all model names (CM/BG/AV → RR/TM)
- [ ] Replace all variable prefixes (cm_/bg_/av_ → rr_/tm_)
- [ ] Replace specialist module class
- [ ] Update specialist_features calculation
- [ ] Update all print statements
- [ ] Update checkpoint names
- [ ] Verify line count (~665 for Stage 1, ~716 for Stage 2, ~777 for Stage 4)

---

## 🎯 COMPLETION ORDER

1. ✅ Complete BG Stage 4 (15 min)
2. ✅ Create RR Stage 1 (15 min)
3. ✅ Create RR Stage 2 (15 min)
4. ✅ Create RR Stage 4 (15 min)
5. ✅ Create TM Stage 1 (15 min)
6. ✅ Create TM Stage 2 (15 min)
7. ✅ Create TM Stage 4 (15 min)

**Total Time:** ~1.5-2 hours

---

## 📊 FEATURE DIMENSIONS REFERENCE

| Model | Specialist Features | Total Features | Adjusted Features |
|-------|-------------------|----------------|-------------------|
| LL    | 68 × 7×7 = 3332   | 1792 + 3332 = 5124 | 5128 (divisible by 8) |
| BG    | 44 × 7×7 = 2156   | 1792 + 2156 = 3948 | 3952 (divisible by 8) |
| AV    | 48 × 7×7 = 2352   | 1792 + 2352 = 4144 | 4144 (divisible by 8) |
| CM    | 40 × 7×7 = 1960   | 1792 + 1960 = 3752 | 3752 (divisible by 8) |
| RR    | 36 × 7×7 = 1764   | 1792 + 1764 = 3556 | 3560 (divisible by 8) |
| TM    | 52 × 7×7 = 2548   | 1792 + 2548 = 4340 | 4344 (divisible by 8) |

---

## 🚀 AFTER COMPLETION

Once all 15 scripts are created:
1. Test each script on Kaggle
2. Train all models sequentially
3. Collect final checkpoints:
   - `ll_model_student.pt` (already trained)
   - `bg_model_student.pt`
   - `av_model_student.pt`
   - `cm_model_student.pt`
   - `rr_model_student.pt`
   - `tm_model_student.pt`
4. Integrate into agent framework
5. Test ensemble prediction

---

## 📁 FILES TO REFERENCE

- **Templates:** `train_cm_stage1_faceforensics.py`, `train_cm_stage2_celebdf.py`, `train_cm_stage4_dfdc.py`
- **Complete Examples:** All AV and CM scripts
- **Module Architectures:** See `FINAL_SESSION_STATUS.md`
- **Status:** `TRAINING_SCRIPTS_STATUS.md`

---

## 💡 TIPS

1. Use Find & Replace (Ctrl+H) for bulk replacements
2. Double-check specialist_features calculation
3. Verify all print statements updated
4. Test import statements work
5. Check line counts match expected (~665, ~716, ~777)

**Good luck with the new session! You're 67% done! 🎉**
