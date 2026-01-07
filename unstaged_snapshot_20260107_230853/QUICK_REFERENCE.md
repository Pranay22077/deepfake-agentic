# 🚀 QUICK REFERENCE CARD

## 📊 STATUS: 10/15 COMPLETE (67%)

### ✅ DONE:
- AV: 3/3 ✅✅✅
- CM: 3/3 ✅✅✅
- BG: 2/3 ✅✅🔄

### ⏳ TODO:
- BG: 1/3 (Stage 4)
- RR: 0/3 (All stages)
- TM: 0/3 (All stages)

---

## 🎯 NEXT 7 TASKS

1. Complete `train_bg_stage4_dfdc.py` (copy from CM Stage 4)
2. Create `train_rr_stage1_faceforensics.py` (copy from CM Stage 1)
3. Create `train_rr_stage2_celebdf.py` (copy from CM Stage 2)
4. Create `train_rr_stage4_dfdc.py` (copy from CM Stage 4)
5. Create `train_tm_stage1_faceforensics.py` (copy from CM Stage 1)
6. Create `train_tm_stage2_celebdf.py` (copy from CM Stage 2)
7. Create `train_tm_stage4_dfdc.py` (copy from CM Stage 4)

---

## 🔧 SPECIALIST FEATURES

| Model | Channels | Features | Status |
|-------|----------|----------|--------|
| LL    | 68       | 3332     | ✅ Done |
| BG    | 44       | 2156     | 🔄 67% |
| AV    | 48       | 2352     | ✅ Done |
| CM    | 40       | 1960     | ✅ Done |
| RR    | 36       | 1764     | ⏳ Todo |
| TM    | 52       | 2548     | ⏳ Todo |

---

## 📝 COPY-PASTE MODULES

### RR Module (36 channels):
```python
class ResolutionModule(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.resolution_analyzer = nn.Sequential(nn.Conv2d(in_channels, 10, 3, padding=1), nn.BatchNorm2d(10), nn.ReLU(), nn.Conv2d(10, 20, 3, padding=1), nn.BatchNorm2d(20), nn.ReLU(), nn.Conv2d(20, 10, 1))
        self.upscaling_detector = nn.Sequential(nn.Conv2d(in_channels, 8, 5, padding=2), nn.BatchNorm2d(8), nn.ReLU(), nn.Conv2d(8, 16, 5, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.Conv2d(16, 8, 1))
        self.edge_checker = nn.Sequential(nn.Conv2d(in_channels, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.Conv2d(16, 8, 1))
        self.interpolation_detector = nn.Sequential(nn.Conv2d(in_channels, 10, 7, padding=3), nn.BatchNorm2d(10), nn.ReLU(), nn.Conv2d(10, 20, 7, padding=3), nn.BatchNorm2d(20), nn.ReLU(), nn.Conv2d(20, 10, 1))
        self.attention = nn.Sequential(nn.Conv2d(36, 32, 1), nn.ReLU(), nn.Conv2d(32, 36, 1), nn.Sigmoid())
    def forward(self, x):
        res = F.adaptive_avg_pool2d(self.resolution_analyzer(x), (7,7))
        ups = F.adaptive_avg_pool2d(self.upscaling_detector(x), (7,7))
        edge = F.adaptive_avg_pool2d(self.edge_checker(x), (7,7))
        interp = F.adaptive_avg_pool2d(self.interpolation_detector(x), (7,7))
        combined = torch.cat([res, ups, edge, interp], dim=1)
        return combined * self.attention(combined)
```

### TM Module (52 channels):
```python
class TemporalModule(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.frame_consistency = nn.Sequential(nn.Conv2d(in_channels, 14, 5, padding=2), nn.BatchNorm2d(14), nn.ReLU(), nn.Conv2d(14, 28, 5, padding=2), nn.BatchNorm2d(28), nn.ReLU(), nn.Conv2d(28, 14, 1))
        self.motion_analyzer = nn.Sequential(nn.Conv2d(in_channels, 12, 7, padding=3), nn.BatchNorm2d(12), nn.ReLU(), nn.Conv2d(12, 24, 7, padding=3), nn.BatchNorm2d(24), nn.ReLU(), nn.Conv2d(24, 12, 1))
        self.temporal_detector = nn.Sequential(nn.Conv2d(in_channels, 12, 3, padding=1), nn.BatchNorm2d(12), nn.ReLU(), nn.Conv2d(12, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(), nn.Conv2d(24, 12, 1))
        self.optical_flow = nn.Sequential(nn.Conv2d(in_channels, 14, 9, padding=4), nn.BatchNorm2d(14), nn.ReLU(), nn.Conv2d(14, 28, 9, padding=4), nn.BatchNorm2d(28), nn.ReLU(), nn.Conv2d(28, 14, 1))
        self.attention = nn.Sequential(nn.Conv2d(52, 32, 1), nn.ReLU(), nn.Conv2d(32, 52, 1), nn.Sigmoid())
    def forward(self, x):
        frame = F.adaptive_avg_pool2d(self.frame_consistency(x), (7,7))
        motion = F.adaptive_avg_pool2d(self.motion_analyzer(x), (7,7))
        temporal = F.adaptive_avg_pool2d(self.temporal_detector(x), (7,7))
        optical = F.adaptive_avg_pool2d(self.optical_flow(x), (7,7))
        combined = torch.cat([frame, motion, temporal, optical], dim=1)
        return combined * self.attention(combined)
```

---

## 🔄 FIND & REPLACE

### For RR:
- `CM` → `RR`
- `cm_` → `rr_`
- `Compression` → `Resolution`
- `CompressionModule` → `ResolutionModule`
- `40 * 7 * 7` → `36 * 7 * 7`

### For TM:
- `CM` → `TM`
- `cm_` → `tm_`
- `Compression` → `Temporal`
- `CompressionModule` → `TemporalModule`
- `40 * 7 * 7` → `52 * 7 * 7`

---

## 📁 TEMPLATES

- **Stage 1:** `train_cm_stage1_faceforensics.py`
- **Stage 2:** `train_cm_stage2_celebdf.py`
- **Stage 4:** `train_cm_stage4_dfdc.py`

---

## ⏱️ TIME ESTIMATE

- BG Stage 4: 15 min
- RR (3 scripts): 45 min
- TM (3 scripts): 45 min
- **Total: ~1.5-2 hours**

---

## 📚 FULL DOCS

Read these for complete details:
1. `CONTINUE_IN_NEW_SESSION.md` ⭐
2. `FINAL_SESSION_STATUS.md` ⭐
3. `SESSION_HANDOFF_SUMMARY.md`

---

**YOU'RE 67% DONE! KEEP GOING! 🚀**
