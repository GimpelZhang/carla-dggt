# E2E Test Manual Verification Checklist

This checklist provides manual verification steps for E2E tests.
Automated tests verify basic functionality; manual checks ensure visual quality
and integration correctness.

## Prerequisites

Before running E2E tests:
- [ ] CARLA server running at localhost:2000
- [ ] DGGT server running at localhost:50051
- [ ] Scene data available at configured path
- [ ] OpenDRIVE map file exists
- [ ] Python environment activated: `conda activate dggt`

## Test E2E-01: Single Frame Render

**Automated checks:**
- Image is not None
- Image shape matches expected resolution
- Image dtype is uint8
- Image mean > 0 (not all black)

**Manual verification:**
- [ ] Run test: `python run_e2e_tests.py --test e2e_01 -v`
- [ ] Check output image saved to test_output/
- [ ] Visual inspection:
  - [ ] Image shows realistic scene rendering
  - [ ] No obvious artifacts or distortion
  - [ ] Colors appear natural
  - [ ] Scene geometry matches expected content

## Test E2E-04: Camera Specifications

**Automated checks:**
- Multiple resolution renders complete
- Ftheta camera renders without errors

**Manual verification:**
- [ ] Run test: `python run_e2e_tests.py --test e2e_04 -v`
- [ ] Check output images for each resolution:
  - [ ] 256x256: image_256x256.jpg
  - [ ] 512x512: image_512x512.jpg
  - [ ] 1024x768: image_1024x768.jpg
- [ ] Visual inspection:
  - [ ] Higher resolution images have more detail
  - [ ] All images show same scene content (different quality)
  - [ ] Ftheta camera shows fisheye distortion pattern

## Pre-Test Setup

1. **Environment setup:**
   ```bash
   source /home/junchuan/miniconda3/etc/profile.d/conda.sh
   conda activate dggt
   cd /mnt/E/carla/carla/PythonAPI/examples/nvidia/nurec
   ```

2. **Start CARLA server:**
   ```bash
   # From CARLA package directory
   ./CarlaUE4.sh
   ```

3. **Start DGGT server:**
   ```bash
   cd /mnt/E/carla/carla/PythonAPI/examples/nvidia/nurec/dggt_server
   python main.py --config ../configs/dggt_server_config.yaml
   ```

4. **Verify servers:**
   ```bash
   # Check CARLA
   python -c "import carla; c=carla.Client('localhost',2000); print(c.get_world())"
   
   # Check DGGT
   python -c "import grpc; from nre.grpc.protos import sensorsim_pb2_grpc; ch=grpc.insecure_channel('localhost:50051'); stub=sensorsim_pb2_grpc.SensorsimServiceStub(ch); print(stub.get_version(...))"
   ```

## Post-Test Cleanup

1. **Stop servers:**
   - CARLA: Ctrl+C in terminal or close window
   - DGGT: Ctrl+C in terminal

2. **Check output directory:**
   ```bash
   ls -la test_output/
   ```

3. **Review logs:**
   ```bash
   # Check for errors in test output
   grep -i error test_output/*.log
   ```

## Troubleshooting

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| CARLA connection timeout | CARLA server not running | Start CarlaUE4.sh |
| DGGT connection refused | DGGT server not running | Start dggt_server/main.py |
| Scene path not found | Wrong path in config | Update e2e_config.yaml |
| OpenDRIVE map not found | XODR file missing | Verify xodr_map_path |
| Render returns None | Scene not loaded | Check DGGT server logs |
| Image all black | Invalid camera pose | Verify pose coordinates |

## Test Execution Log Template

```
Date: ______
Operator: ______
CARLA Version: ______
DGGT Version: ______

E2E-01:
- Start time: ______
- End time: ______
- Status: PASS/FAIL
- Notes: ______

E2E-04:
- Start time: ______
- End time: ______
- Status: PASS/FAIL
- Notes: ______
```

## References

- Phase6_Detailed_Plan.md: Full E2E test specification
- README.md: Integration documentation
- dggt_integration.py: Main integration module