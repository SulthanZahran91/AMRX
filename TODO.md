# AMRx Simulator - Implementation TODO

## Current Status
**Phase**: Initial Setup - Awaiting clarifications before implementation

## Questions to Resolve
(See CLAUDE.md for responses)

---

## Implementation Phases

### Phase 1: Core Kinematics (Days 1-2)
- [ ] Set up project structure
- [ ] Create Robot class with basic update(v, ω, dt)
- [ ] Implement coordinate system and angle normalization
- [ ] Implement differential drive kinematics (Euler integration)
- [ ] Basic visualization with Pygame
- [ ] Unit tests for kinematics
- [ ] Validate kinematic equations

### Phase 2: World Geometry (Day 3)
- [ ] Create World class
- [ ] Implement wall representation
- [ ] Implement landmark (cylinder) representation
- [ ] Load world from JSON file
- [ ] Implement collision detection (circle-line segment)
- [ ] Test with box environment
- [ ] Unit tests for collision detection

### Phase 3: LiDAR Sensor (Days 4-5)
- [ ] Create LidarSensor class
- [ ] Implement ray-line segment intersection
- [ ] Implement ray-circle intersection
- [ ] Implement full 2D scan (361 rays)
- [ ] Add measurement noise
- [ ] Visualize rays hitting walls
- [ ] Optimize with spatial grid partitioning (if needed)
- [ ] Unit tests for ray intersections

### Phase 4: Odometry Sensor (Day 6)
- [ ] Create OdometrySensor class
- [ ] Implement error model (systematic + random)
- [ ] Output in robot frame format
- [ ] Compare dead reckoning vs ground truth
- [ ] Validate drift accumulation
- [ ] Unit tests for odometry

### Phase 5: Landmark Detector (Day 7)
- [ ] Create LandmarkDetector class
- [ ] Implement visibility criteria (distance, bearing, FOV)
- [ ] Implement line-of-sight occlusion check
- [ ] Add measurement noise (range, bearing)
- [ ] Add to visualization
- [ ] Unit tests for landmark detection

### Phase 6: Robot Dynamics (Day 8)
- [ ] Implement slew rate limiting
- [ ] Add max velocity constraints
- [ ] Add max acceleration constraints
- [ ] Test step response
- [ ] Unit tests for dynamics

### Phase 7: Data Logging (Day 9)
- [ ] Implement HDF5 logging
- [ ] Create buffered writing strategy
- [ ] Log ground truth data
- [ ] Log sensor data
- [ ] Create plotting/analysis scripts
- [ ] Test log file integrity

### Phase 8: Simulation Control (Day 9-10)
- [ ] Implement real-time mode
- [ ] Implement headless/fast mode
- [ ] Implement deterministic behavior (random seed)
- [ ] Create clean API (step, reset, get_ground_truth, close)
- [ ] Time management and synchronization

### Phase 9: Testing & Validation (Day 10)
- [ ] Complete unit test suite
- [ ] Integration tests (dead reckoning drift)
- [ ] Integration tests (LiDAR in empty box)
- [ ] Create example scripts
- [ ] Performance testing

### Phase 10: Polish & Documentation (Day 10)
- [ ] Code documentation
- [ ] Usage examples
- [ ] README with quickstart
- [ ] Performance optimization if needed
- [ ] Final validation against specification

---

## Completed Tasks
(None yet)

---

## Current Blockers
(None yet)

---

## Notes
- Following implementation priority order from spec Section 15
- Will commit after each phase for version control
