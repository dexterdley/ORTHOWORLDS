### Humanoid & Creature Locomotion
1. Mountain Goat 🐐

- Reward Objective: Maximize forward distance traveled upward while minimizing falls or sliding backward down the procedural cliff.
- Action Space: Torso pitch, roll, yaw; hip and leg joint torques/angles.

2. Spider Walker

- Reward Objective: Maximize forward velocity of the main body while maintaining rhythmic, stable leg cycles and penalizing torso drops.
- Action Space: Multi-leg joint torques (hips, knees, ankles).

3. Parkour Runner

- Reward Objective: Maximize checkpoints cleared and speed, with bonuses for clean vaulting over obstacles and penalties for tripping or falling into gaps.
- Action Space: Bipedal leg joint torques, torso leaning, jump impulse.

4. Ice Skater

- Reward Objective: Maintain high forward momentum and balance along a curved track without wiping out or sliding off-course.
- Action Space: Skate blade edge angles, leg push torques, upper body sway.

5. Kangaroo Hopper

- Reward Objective: Maximize distance covered per jump while minimizing energy expended (joint torque usage).
- Action Space: Tail pitch control, hip/knee/ankle spring extension force.

6. Tightrope Walker

- Reward Objective: Keep the center of mass directly above the rope for as long as possible while moving forward to the end goal.
- Action Space: Balancing pole tilt angle, foot placement pitch/roll, ankle torques.

7. Lizard Climber

- Reward Objective: Maximize vertical height climbed on a sheer wall using alternating sticky grip contacts.
- Action Space: Limb extension vectors, adhesive pad engagement states, spine lateral flexion.

8. Monkey Swing

- Reward Objective: Maximize forward distance traveled across suspended anchor points using momentum and arm releases.
- Action Space: Shoulder/elbow joint torques, left/right hand grip release timing.

9. Penguin Slide

- Reward Objective: Maximize distance traveled by efficiently alternating between walking uphill and belly-sliding down slopes.
- Action Space: Flipper thrust, toboggan posture switch, steering tilt.

10. Robot Acrobat

- Reward Objective: Successfully execute specific aerial rotations (flips/spins) and land cleanly on both feet within a target zone.
- Action Space: Mid-air body tuck/extend joint torques, landing leg damping.

11. Centipede March

- Reward Objective: Maximize forward crawl speed of the head segment while keeping body segments aligned in a coordinated wave.
- Action Space: Segment-wise leg pair actuation phases, head steering angle.

12. Frog Escape

- Reward Objective: Maximize distance crossed across unstable, floating lily-pad platforms without falling into the water.
- Action Space: Hind leg compression charge, launch angle pitch/yaw.

13. Mars Walker

- Reward Objective: Stabilize bipedal locomotion across low-gravity, uneven terrain to reach a designated outpost.
- Action Space: Ankle/knee/hip actuator torques, waist stabilization pitch.

14. Exoskeleton Pilot

- Reward Objective: Match a target walking gait and carry heavy payload boxes across a factory floor without dropping them.
- Action Space: Servo assistance torques, arm clamp pressure, gait stride frequency.

15. Ragdoll Rescue

- Reward Objective: Drag or carry an unstable, floppy body safely to an exit zone within a time limit.
- Action Space: Gripper position, hoist tension, vehicle/agent locomotion vector.


### Vehicle Games
1. Monster Truck Trial

- Reward Objective: Complete the obstacle course as fast as possible without flipping the truck or breaking suspension components.
- Action Space: Throttle/brake force, front/rear steering angles, active suspension damping.

2. Delivery Truck

- Reward Objective: Transport fragile cargo to the destination finish line with zero package loss and minimal transit time.
- Action Space: Steering angle, acceleration/braking torque, cargo tie-down tension.

3. Tank Commander

- Reward Objective: Navigate rough battlefield terrain to reach a designated waypoint while managing tread traction.
- Action Space: Left tread velocity, right tread velocity, turret rotation angle.

4. Steamroller

- Reward Objective: Maximize the percentage of random surface objects and debris flattened within a set timeframe.
- Action Space: Forward/reverse drive torque, roller steering angle, compaction weight force.

5. Mining Rover

- Reward Objective: Collect and haul high-value mineral chunks back to a refinery station in rugged cave systems.
- Action Space: 6-wheel independent drive torques, scoop arm articulation, tilt angle.

6. Moon Buggy

- Reward Objective: Drive across lunar craters at maximum speed without crashing or flipping during low-gravity airtime.
- Action Space: Wheel drive torque, pitch thruster control during jumps, suspension spring stiffness.

7. Hovercraft Racer

- Reward Objective: Navigate a slippery, frictionless swamp track through checkpoint gates as fast as possible.
- Action Space: Rear thrust magnitude, left/right directional rudder vanes, lift fan power.

8. Desert Rally

- Reward Objective: Cross shifting sand dunes to reach checkpoints before fuel depletion.
- Action Space: 4WD torque distribution, tire pressure control, steering angle.

9. Snowmobile Explorer

- Reward Objective: Climb steep, slippery icy hills while avoiding deep snow pits and crevasses.
- Action Space: Track drive throttle, ski steering angle, rider body weight shift.

10. Unicycle Rider

- Reward Objective: Maintain balance and pedal forward continuously along a bumpy surface profile.
- Action Space: Pedal torque, seat post pitch/roll tilt, arm sway balance.

11. Forklift Logistics

- Reward Objective: Pick up heavy stacked crates and place them accurately onto designated grid locations.
- Action Space: Rear wheel steering angle, drive throttle, fork vertical lift, fork mast tilt.

12. Farm Tractor

- Reward Objective: Pull a heavy, sliding trailer load through deep mud trenches to the end of a farm lane.
- Action Space: Engine RPM/gear ratio, differential lock engagement, steering torque.

13. Train Builder

- Reward Objective: Connect and manage multiple heavy cargo wagons along a winding track without derailing on corners.
- Action Space: Locomotive throttle, braking force per car, track switch lever signals.

14. Rescue Ambulance

- Reward Objective: Navigate city traffic and rough obstacles to reach victims quickly with minimal structural g-force shock.
- Action Space: Steering angle, gas/brake pedal pressure, siren alert pulse.

15. Bike Courier

- Reward Objective: Deliver packages across tight urban street obstacles within a strict delivery deadline.
- Action Space: Handlebar steer angle, pedal cadence force, front/rear brake levers.


### Space & Sci-Fi
1. Asteroid Miner

- Reward Objective: Use thrusters to dock with tumbling space rocks, harvest ore, and return to base.
- Action Space: Main thruster force, RCS translation/attitude torques, laser drill power.

2. Satellite Docking

- Reward Objective: Align a spacecraft's position and orientation precisely with a docking port using minimal RCS fuel.
- Action Space: 6-DOF RCS translation (X, Y, Z) and rotation (pitch, roll, yaw) pulses.

3. Orbital Tugboat

- Reward Objective: Push dead satellites and heavy orbital cargo modules into a stable parking orbit.
- Action Space: Directional thrust vectoring, grapple cable winch tension, main engine throttle.

4. Alien Terrain Explorer

- Reward Objective: Drive across unknown, low-traction planetary crust while mapping hazards.
- Action Space: All-wheel drive torque, suspension height adjustment, scanner turret angle.

5. Rocket Landing Challenge

- Reward Objective: Perform a vertical retro-rocket touchdown on a landing pad with zero lateral drift and soft velocity.
- Action Space: Main engine thrust vectoring/gimbal, cold-gas attitude thrusters, landing leg deployment.

6. Drone Swarm

- Reward Objective: Keep multiple autonomous drones in a tight geometric formation while navigating around obstacles to a target.
- Action Space: Individual quadrotor thrust levels (4 per drone) across the fleet.

7. Gravity Shift

- Reward Objective: Navigate a maze as environmental gravity direction rotates dynamically at set intervals.
- Action Space: Agent propulsion force, magnetic boots grip toggle, orientation thrusters.

8. Wormhole Navigator

- Reward Objective: Conserve and manipulate spacecraft momentum to thread through twisting portal rings.
- Action Space: Pitch/yaw attitude control, burst impulse engine, shield phase angle.

9. Magnetic Rover

- Reward Objective: Switch magnetic polarity to stick to metallic ceilings and walls to bypass bottom hazards.
- Action Space: Wheel drive torque, electromagnetic attraction strength per wheel, polarity inversion trigger.

10. Space Debris Cleanup

- Reward Objective: Grapple and clear floating junk fragments out of an orbital lane within a time limit.
- Action Space: Robotic arm joint angles, harpoon launcher angle/force, debris collection bay door.

11. Laser Mining Bot

- Reward Objective: Use cutting beams to slice specific mineral deposits off procedural cave walls and catch them.
- Action Space: Hover thrusters, beam orientation pitch/yaw, intensity regulator, collector basket position.

12. Planet Hopper

- Reward Objective: Launch from the gravitational pull of one small mini-planet to intercept and land on another.
- Action Space: Launch impulse angle and velocity, mid-course corrective micro-thrusters.

13. Fuel Economy Mission

- Reward Objective: Maximize total distance traveled across deep space with a strictly limited fuel reserve.
- Action Space: Burn timing ignition/duration, burn angle vector, gravity-assist sling angles.

14. Escape Velocity

- Reward Objective: Time and angle rocket booster ignition to successfully break free from a high-gravity planet's surface.
- Action Space: Multi-stage staging triggers, pitch program tilt schedule, throttle percentage.

15. Anti-Gravity Racer

- Reward Objective: Maintain extreme speeds on floating, loop-heavy tracks without wall collisions.
- Action Space: Repulsor pitch/roll air-brakes, main plasma engine thrust, magnetic track lock force.


### Construction & Manipulation
1. Crane Operator

- Reward Objective: Swing and lower heavy construction beams precisely into blueprint socket zones.
- Action Space: Tower slewing rotation, trolley boom position, winch cable drop/lift velocity.

2. Bridge Builder RL

- Reward Objective: Arrange structural beams to support a weight load crossing a gap without structural collapse.
- Action Space: Node connection placement coordinates, beam material type, joint tension lock.

3. Excavator Simulator

- Reward Objective: Scoop loose dirt/rocks with the bucket and dump them accurately into a target container.
- Action Space: Tracks drive torque, cabin swing angle, boom/dipper/bucket hydraulic cylinder speeds.

4. Tower Stacker

- Reward Objective: Balance uneven blocks on top of each other to build the tallest possible stable structure.
- Action Space: Overhead claw horizontal X/Y displacement, block release trigger, rotation angle.

5. Demolition Expert

- Reward Objective: Place and trigger explosive charges to collapse a multi-story building entirely within its footprint.
- Action Space: Charge placement coordinates (X, Y, Z), delay timer settings, detonation sequence trigger.

6. Warehouse Robot

- Reward Objective: Sort and transport mixed packages to designated color-coded shelves efficiently.
- Action Space: Differential drive wheel speeds, scissor lift height, suction gripper activation.

7. Magnet Crane

- Reward Objective: Lift and sort scattered metallic scraps into recycling bins using electromagnet pulses.
- Action Space: Gantry crane X/Y carriage velocity, magnet height, coil current ON/OFF.

8. Pipeline Repair

- Reward Objective: Rotate and slide heavy pipe segments together to seal a leaking fluid line.
- Action Space: Manipulator arm 6-axis joint velocities, clamp locking force, gasket alignment.

9. Container Port

- Reward Objective: Unload shipping containers from a swaying cargo ship onto dock transport trucks.
- Action Space: Spreader bar alignment twist, gantry trolley speed, hoist cable tension.

10. Castle Builder

- Reward Objective: Assemble stone blocks into a fortified wall capable of withstanding external impacts.
- Action Space: Masonry arm end-effector pose, mortar spray toggle, stone block selection index.

11. Road Constructor

- Reward Objective: Lay down smooth paving segments across deep chasms and uneven hills.
- Action Space: Asphalt paver drive speed, screed height/vibration, grade slope angle.

12. Factory Automation

- Reward Objective: Route moving components across conveyor belts into correct processing machines without jams.
- Action Space: Diverter arm angles, belt speed controllers, pneumatic pusher pulses.

13. Lumberjack Machine

- Reward Objective: Cut down procedural tree trunks and load the logs onto a flatbed trailer.
- Action Space: Harvester head clamp pressure, saw chain feed speed, boom extension/tilt.

14. Wrecking Ball

- Reward Objective: Swing a heavy cable ball to smash targeted structural pillars down to a specified height.
- Action Space: Crane cab rotation velocity, boom elevation angle, cable pay-out/spool speed.

15. Bridge Inspector

- Reward Objective: Drive a multi-wheeled crawler across decaying, unstable bridge trusses without falling through.
- Action Space: Wheel speeds, arm sensor probe position, structural stabilizer leg deployment.


### Survival & Hazard Environments
1. Volcano Escape

- Reward Objective: Outrun rising lava flows and dodge falling volcanic debris chunks.
- Action Space: Steering angle, drive acceleration/brake, jump/thruster booster.

2. Avalanche Survivor

- Reward Objective: Maneuver down a mountain slope ahead of a massive, rolling snowpack wave.
- Action Space: Carving edge angle, tuck/stand posture, jump impulse.

3. Flood Rescue

- Reward Objective: Rescue stranded agents or secure items before floodwaters submerge the play area.
- Action Space: Airboat throttle, rudder angle, winch rescue line reel speed.

4. Earthquake Run

- Reward Objective: Sprint or drive across terrain that is actively fracturing and shifting apart.
- Action Space: Directional movement vector, jump over chasm trigger, speed control.

5. Toxic Gas Escape

- Reward Objective: Find the fastest exit path out of a labyrinth before a gas cloud fills the space.
- Action Space: Drive force (forward/reverse), steering torque.

6. Forest Fire Evacuation

- Reward Objective: Navigate vehicles away from dynamically spreading wildfire zones and burning timber.
- Action Space: Vehicle steering angle, throttle, water cannon spray direction.

7. Meteor Dodge

- Reward Objective: Keep a mobile agent alive within a zone bombarded by random falling space debris.
- Action Space: 2D ground movement vector (X, Y), dodge roll impulse.

8. Storm Chaser

- Reward Objective: Drive close to swirling vortex hazards to collect atmospheric data while avoiding structural damage.
- Action Space: Vehicle steering angle, throttle, deploy sensor anchor spike.

9. Minefield Crossing

- Reward Objective: Step carefully or drive across a terrain field embedded with hidden explosive triggers.
- Action Space: Step location X/Y, probe pressure sensitivity, vehicle tread speed.

10. Radiation Zone

- Reward Objective: Plan a path that minimizes cumulative exposure time to radioactive hot spots while reaching safety.
- Action Space: Rover movement velocity, shielding panel orientation angle.


### Puzzle & Planning
1. Lever Puzzle

- Reward Objective: Activate mechanical switches in the correct logical sequence to open a heavy gateway.
- Action Space: Push force vector, lever interaction toggle, movement position.

2. Chain Reaction

- Reward Objective: Position dominoes and physical triggers so a single push causes all targets to activate.
- Action Space: Object placement coordinates, rotation angle, initial push force vector.

3. Key and Door

- Reward Objective: Solve physics-based obstacles to retrieve a physical key and bring it to a locked door.
- Action Space: Agent movement force, object pickup/drop, key alignment angle.

4. Weighted Platforms

- Reward Objective: Place exact weights onto scale platforms to balance a complex lever bridge.
- Action Space: Weight block selection, grabber arm position, release trigger.

5. Bridge Timing

- Reward Objective: Time movement across retracting or rotating bridges to cross safely without falling.
- Action Space: Forward velocity, stop/go brake, sprint burst trigger.

6. Pendulum Crossing

- Reward Objective: Jump between swinging cargo hooks at optimal moments to cross a chasm.
- Action Space: Swing momentum pump force, release grip timing, jump launch angle.

7. Ball Routing

- Reward Objective: Arrange static ramps and bumpers to guide a rolling ball into a distant goal cup.
- Action Space: Ramp placement X/Y, ramp tilt angle, bumper elasticity setting.

8. Physics Maze

- Reward Objective: Tilt the entire world container to roll a ball through a labyrinth to the exit.
- Action Space: World container pitch angle, world container roll angle.

9. Energy Management

- Reward Objective: Allocate limited electrical grid power to actuators to keep systems running under load.
- Action Space: Power routing switches (0-100% per sub-system), circuit breaker toggles.

10. Signal Activation

- Reward Objective: Trigger pressure plates and optical sensors in a precise temporal order.
- Action Space: Agent movement velocity, sensor beam reflector angle.


### Combat & Competitive Physics
1. Robot Sumo

- Reward Objective: Apply force to push opponent wrestling robots outside the boundary ring while staying inside.
- Action Space: Left/right wheel drive torques, plow wedge lift angle.

2. Tank Duel

- Reward Objective: Aim turret angles, fire projectiles, and evade incoming enemy fire.
- Action Space: Tank drive velocity, turret rotation angle, cannon elevation, fire trigger.

3. Capture the Flag

- Reward Objective: Retrieve the opponent's physical flag and bring it back to your base zone.
- Action Space: Vehicle/agent movement vector, flag carrier tackle/bump, boost.

4. Arena Survival

- Reward Objective: Defend position against waves of incoming dynamic hazards or aggressive entities.
- Action Space: Shield direction angle, movement impulse, shockwave blast trigger.

5. Catapult War

- Reward Objective: Adjust tension and launch angles to destroy enemy structures with heavy boulders.
- Action Space: Winch tension force, arm launch angle, trigger release lock.

6. Turret Defender

- Reward Objective: Angle defensive shields and projectile launchers to block incoming barrage objects.
- Action Space: Shield rotation angle, turret pitch/yaw, interceptor missile launch.

7. Mech Battle

- Reward Objective: Control multi-limbed bipedal warbots to destabilize and knock down opposing fighters.
- Action Space: Torso twist angle, leg joint torques, arm punch/block vectors.

8. Drone Dogfight

- Reward Objective: Outmaneuver opposing aerial units in a closed physics dogfighting arena.
- Action Space: East-West thrust (+X), North-South thrust (+Y), weapon fire trigger.

9. Physics Soccer

- Reward Objective: Use a vehicle or agent to maneuver a giant, heavy ball past defenders into the goal net.
- Action Space: Vehicle drive throttle, steering angle, flip/boost hit impulse.

10. King of the Hill

- Reward Objective: Fight off competitors to maintain control of an elevated center platform.
- Action Space: Agent movement force, push/shove bumper force, anchor stability stance.


### Novel Research Benchmark Environments
1. Adaptive Gravity World

- Reward Objective: Continuously adapt movement policies as environmental gravity magnitude and direction shift unpredictably.
- Action Space: Multi-directional thrusters, leg joint adaptation torques, suction grip.

2. Morphing Terrain

- Reward Objective: Navigate across ground geometry that warps, expands, and collapses underneath the agent in real-time.
- Action Space: Leg stride frequency, body clearance elevation, jump impulse.

3. Damage-Aware Robot

- Reward Objective: Complete tasks successfully even when random limbs or joints suffer structural failures mid-episode.
- Action Space: Remaining joint actuators torques, weight redistribution posture.

4. Tool Discovery

- Reward Objective: Experiment with loose physical objects in the environment to discover and use them as levers or bridges.
- Action Space: Arm end-effector 3D translation, gripper open/close, push force.

5. Multi-Body Assembly

- Reward Objective: Grab scattered loose parts and collide them together correctly to snap a complex structure into place.
- Action Space: Robotic gripper 6-DOF pose, alignment rotation, snap-fit press force.

6. Procedural Ecosystem

- Reward Objective: Manage energy or survive in a dynamic environment populated by interacting mobile entities.
- Action Space: Locomotion direction/speed, resource ingestion toggle, defense posture.

7. Delayed Reward Expedition

- Reward Objective: Successfully navigate a long, complex maze where zero intermediate feedback is given until the final destination.
- Action Space: Steering angle, drive speed, waypoint marker drop.

8. Curriculum Planet

- Reward Objective: Adapt to a world where environmental physics parameters scale up in difficulty continuously.
- Action Space: Wheel drive torques, suspension stiffness, active wing downforce.

9. Open-World Collector

- Reward Objective: Maximize rare item discovery and map exploration across a vast sparse-reward layout.
- Action Space: Vehicle/agent movement vector, item scanner pulse, boost.

10. Meta-Physics Challenge

- Reward Objective: Infer and adapt to completely random physical constants (friction, restitution, mass) changing at the start of each episode.
- Action Space: Probe force impulse, adaptive gait/drive torques.