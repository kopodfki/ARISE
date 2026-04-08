# load the eval prompt
import os
import re
import random
import datetime
from cv2 import accumulate
from google import genai
from google.genai import types
# Initialize the GenAI client

with open('eval_prompt.txt', 'r') as file:
    eval_prompt = file.read()

with open('list_of_scenarios.txt', 'r') as file:
    list_of_scenarios = file.read().splitlines()

REF_SCENARIOS_FOLDER = 'reference_scenarios/'
SCENARIOS_FOLDER = 'eval_test_set/gemini'

scenario_folders = [f for f in os.listdir(
    SCENARIOS_FOLDER) if os.path.isdir(os.path.join(SCENARIOS_FOLDER, f))]


def generate(message):
    client = genai.Client(
        api_key="AIzaSyBZlz5DoRVhDaQUM5t8UeKjjuqpVffYzXk",
    )

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""Here is a shortened version of scenic 3.0 documentation that is relevant to the evaluation of scenarios:
                    ## Scenic 3.0: Syntax, Structure, and Spatial Relations

                    Scenic is a domain-specific probabilistic programming language designed for modeling and generating scenarios for cyber-physical systems, particularly in robotics and autonomous driving. It allows users to define distributions over scenes (static configurations of objects) and dynamic policies (how agents act over time), enabling the generation of diverse training and testing data for simulators.

                    **Key takeaway for version 3.0:** It introduces native **3D geometry**, precise **object shapes**, and enhanced **temporal requirements**, with significant syntax and semantic changes from previous versions (e.g., requiring the `new` keyword for object instantiation, and changes to `heading` property derivation).

                    ---

                    ### 1. Scenic Syntax Overview

                    Scenic's syntax is inspired by Python, but extends it with specialized keywords and operators for scenario definition.

                    *   **Object Instantiation:**
                        *   The `new` keyword is now **required** for creating instances of classes (e.g., `ego = new Object`). This resolves ambiguity from earlier versions.
                        *   Objects are defined with a set of `specifiers` (see Spatial Relations below).
                            ```scenic
                            ego = new Object with shape ConeShape(),
                                        with width 2,
                                        with height 1.5,
                                        facing (-90 deg, 45 deg, 0)
                            ```

                    *   **Class Definitions:**
                        *   Similar to Python classes, supporting inheritance and methods.
                        *   Properties can be defined with default values, which can depend on other properties of `self`.
                            ```scenic
                            class Vehicle:
                                pass

                            class Car:
                                position: new Point on road
                                heading: roadDirection at self.position
                                width: self.model.width
                            ```

                    *   **Distributions:** Scenic is a probabilistic language, so random values are defined using distributions.
                        *   `Range(low, high)`: Uniformly-distributed real number.
                        *   `DiscreteRange(low, high)`: Uniformly-distributed integer.
                        *   `Normal(mean, stdDev)`: Normal distribution.
                        *   `TruncatedNormal(mean, stdDev, low, high)`: Truncated normal distribution.
                        *   `Uniform(value1, value2, ...)`: Uniformly selects from a finite set of values.
                        *   `Discrete({value: weight, ...})`: Discrete distribution with given weights.

                    *   **Requirements/Constraints (`require`):**
                        *   Define conditions that must be satisfied by generated scenes or simulations.
                        *   **Hard Requirements:** `require boolean_expression` (e.g., `require car2 can see ego`). If violated, the scene/simulation is rejected.
                        *   **Soft Requirements:** `require[probability] boolean_expression` (e.g., `require[0.5] car2 can see ego`).
                        *   **Temporal Requirements:** Use temporal operators within `require` for dynamic scenarios:
                            *   `always condition`
                            *   `eventually condition`
                            *   `next condition`
                            *   `condition until condition`
                            *   `hypothesis implies conclusion`

                    *   **Behaviors (`behavior`):** Define dynamic policies for agents.
                        *   Functions that execute actions over time using `take action` or `wait`.
                        *   Can have `precondition` and `invariant` guards.
                        *   Control flow (`if`, `while`) inside behaviors can depend on random values (unlike top-level code).
                            ```scenic
                            behavior FollowLaneBehavior():
                                while True:
                                    # compute controls
                                    take SetThrottleAction(throttle), SetSteerAction(steering)
                            ```

                    *   **Monitors (`monitor`):** Run in parallel with scenarios, typically for checking properties without taking actions.
                        *   Can maintain state (local variables).
                        *   Instantiated using `require monitor monitor_name(args)`.

                    *   **Modular Scenarios (`scenario`):** Define reusable scenario components.
                        *   `setup:` block: executed once at compilation, defines objects, requirements.
                        *   `compose:` block: orchestrates execution of other scenarios/behaviors over time using `do`.
                        *   Composition statements: `do scenario`, `do scenario until condition`, `do scenario for N seconds/steps`, `do choose scenario1, scenario2`, `do shuffle scenario1, scenario2`.

                    *   **Mutations (`mutate`):** Randomly vary properties of existing objects for testing.
                        *   `mutate object_list [by scalar_factor]`

                    *   **Recording (`record`):** Save values during simulation for analysis.
                        *   `record value as name` (at every time step).
                        *   `record initial value as name` (at start).
                        *   `record final value as name` (at end).

                    *   **Interrupts (`try: ... interrupt when ...`):** Allows behaviors/scenarios to suspend and execute an interrupt handler when a condition is met. `abort` can terminate the `try-interrupt` block.

                    ---

                    ### 2. Scenic Structure

                    Scenic programs define a **probability distribution** over possible scenes and dynamic evolutions. The overall process involves:

                    1.  **Compilation:** A Scenic program is compiled into a `Scenario` object. This involves parsing the Scenic code into an Abstract Syntax Tree (AST), transforming it into a Python AST, and executing it. During this phase, distributions and requirements are set up.
                    2.  **Scene Generation (Sampling):** From the `Scenario` object, concrete `Scene` objects are sampled. This is a rejection sampling process: Scenic makes random choices (from distributions) and then checks if the resulting scene satisfies all *hard requirements*. If not, the sample is rejected, and a new one is tried. Pruning techniques help make this more efficient by avoiding infeasible parts of the sample space.
                        *   A `Scene` object contains: `objects` (physical objects, including `egoObject`), `params` (global parameters), and `workspace` (the overall bounding region).
                    3.  **Dynamic Simulation (Execution):** For dynamic scenarios, a `Simulator` is used to run the `Scene` over time. The behaviors of agents run in parallel, taking actions at discrete time steps. Requirements (including temporal ones and monitor checks) are continuously evaluated, and violations lead to rejection of the simulation.

                    **Hierarchical Object Model:**
                    Scenic provides a built-in hierarchy for defining objects:
                    *   **`Point`**: Basic spatial location (3D coordinates: `(x, y, z)`). Default Z is 0 for 2D compatibility.
                    *   **`OrientedPoint`**: Extends `Point` with an `orientation`. Orientation is a 3D quaternion, derived from `parentOrientation` and intrinsic `yaw`, `pitch`, `roll` Euler angles.
                    *   **`Object`**: Extends `OrientedPoint` with physical properties:
                        *   `width`, `length`, `height` (dimensions of its bounding box).
                        *   `shape` (e.g., `BoxShape`, `ConeShape`, `MeshShape` loaded from STL files).
                        *   `allowCollisions` (bool, default `False`: objects don't overlap).
                        *   `regionContainedIn` (the `Region` the object must be within, default `workspace`).
                        *   `requireVisible` (bool, default `False`: object must be visible from `ego`).
                        *   `behavior` (dynamic policy for agents).
                        *   `velocity`, `speed`, `angularVelocity`, `angularSpeed` (dynamic state).

                    ---

                    ### 3. Spatial Relations in Depth

                    Spatial relations are a core strength of Scenic, allowing for natural language-like descriptions of how objects are positioned relative to each other and the environment. This is achieved through powerful **specifiers** and **operators**.

                    #### Specifiers (defining object properties at creation):

                    These are used after `new ClassName` to define its properties. They often interact with each other and are resolved by Scenic using a priority system.

                    *   **Absolute Positioning:**
                        *   `at vector`: Places the object at global coordinates.
                            ```scenic
                            new Object at (10, 5, 2)
                            ```
                    *   **Relative Positioning (to `ego` by default, or another object/point):**
                        *   `offset by vector`: Position relative to the `ego`'s local coordinate system.
                            ```scenic
                            new Object offset by (5, 0, 0) # 5 units ahead of ego
                            ```
                        *   `offset along direction by vector`: Position relative to a given direction.
                        *   `(left | right) of (vector | OrientedPoint | Object) [by scalar]`: Positions to the left/right. The `by` scalar defines the distance between bounding boxes.
                            ```scenic
                            new Car left of ego by 2 # 2 units to the left of ego's bounding box
                            ```
                        *   `(ahead of | behind) (vector | OrientedPoint | Object) [by scalar]`: Positions ahead/behind.
                        *   `(above | below) (vector | OrientedPoint | Object) [by scalar]`: Positions above/below.
                        *   `following vectorField [from vector] for scalar`: Positions by following a vector field (e.g., a road).

                    *   **Region-Based Positioning:** Scenic's `Region` objects are fundamental for spatial constraints.
                        *   `in region`: Places object uniformly at random *within* a specified `Region`.
                            ```scenic
                            new Pedestrian in sidewalkRegion # Randomly placed on a sidewalk
                            ```
                        *   `on (region | Object | vector)`: Places the *base* of the object uniformly at random on a surface (e.g., `on floor`, `on road`). This specifier also *modifies* existing positions by projecting them onto the region/surface.
                            ```scenic
                            new Rock on MarsGround # Randomly placed on the Mars ground
                            ```
                        *   `contained in region`: Ensures the *entire object* (not just center/base) is contained within a region.

                    *   **Orientation Specifiers:**
                        *   `facing orientation`: Sets the object's global orientation directly.
                        *   `facing vectorField`: Orients the object along the direction of a vector field at its position (e.g., `facing roadDirection`).
                        *   `facing (toward | away from) vector`: Orients the object to face toward/away from a given point.
                        *   `facing directly (toward | away from) vector`: Sets both yaw and pitch to face a point.
                        *   `apparently facing heading [from vector]`: Orients the object based on its apparent heading relative to `ego`'s line of sight.

                    #### Operators (querying spatial relationships):

                    These are functions that return values based on spatial relationships between objects or points.

                    *   **Scalar Operators (distances, angles):**
                        *   `distance [from vector] to vector`: Euclidean distance.
                        *   `angle [from vector] to vector`: Azimuthal angle.
                        *   `altitude [from vector] to vector`: Vertical distance.
                        *   `relative heading of heading [from heading]`: Relative heading.
                        *   `apparent heading of OrientedPoint [from vector]`: Apparent heading from `ego`.

                    *   **Boolean Operators (conditions):**
                        *   `(Point | OrientedPoint | Object) can see (vector | Object)`: Checks visibility, **accounting for occlusion and 3D shapes** (a major 3.0 feature).
                        *   `(vector | Object) in region`: Checks if a point/object is contained within a region.
                        *   `(Object | region) intersects (Object | region)`: Checks for overlap between shapes/regions.

                    *   **Region Operators (new regions):**
                        *   `visible region`: Returns the portion of a region visible from `ego` (or other point).
                        *   `not visible region`: Returns the portion not visible.

                    *   **OrientedPoint Operators (accessing object parts):**
                        *   `(front | back | left | right | top | bottom) of Object`: Returns an `OrientedPoint` at the midpoint of a specific side of an object's bounding box.
                        *   Combinations like `TopFrontLeft of Object`.

                    ---
                    """),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""I'm also providing you a list of description scenario pairs that you can use as a scenic code reference:
================================================================================
SCENARIO 1
================================================================================

DESCRIPTION:
The ego vehicle is driving on a straight road when a pedestrian suddenly crosses from the right front and suddenly stops as the ego vehicle approaches.

SCENIC CODE:
# BEGIN TOWN
Town = 'Town05'
# END TOWN
# BEGIN IMPORTS
param map = localPath(f'../../assets/maps/CARLA/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
# END IMPORTS
# BEGIN BEHAVIOR
behavior AdvBehavior():
    do CrossingBehavior(ego, globalParameters.OPT_ADV_SPEED, globalParameters.OPT_ADV_DISTANCE, "right") until self in ego.lane or self in ego.lane.successor
    while True:
        take SetWalkingSpeedAction(0)

param OPT_ADV_SPEED = Range(0, 5)
param OPT_ADV_DISTANCE = Range(0, 15)
param OPT_STOP_DISTANCE = Range(0, 1)
# END BEHAVIOR
# BEGIN GEOMETRY
lane = Uniform(*network.lanes)
EgoTrajectory = lane.centerline
EgoSpawnPt = new OrientedPoint on lane.centerline

ego = new Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# END GEOMETRY
# BEGIN SPAWN
param OPT_GEO_X_DISTANCE = Range(2, 8)
param OPT_GEO_Y_DISTANCE = Range(15, 50)

IntSpawnPt = new OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE
AdvAgent = new Pedestrian right of IntSpawnPt by globalParameters.OPT_GEO_X_DISTANCE,
    with heading IntSpawnPt.heading - 90 deg,  # Heading perpendicular to the road, adjusted for left crossing
    with regionContainedIn None,
    with behavior AdvBehavior()
# END SPAWN
# BEGIN REQUIREMENTS
require distance from ego to AdvAgent > 5
require eventually AdvAgent in network.drivableRegion
require AdvAgent in network.walkableRegion
require eventually ego.canSee(AdvAgent, occludingObjects=tuple([])) # Make sure to replace the empty list with a list of all other simulation agents
# END REQUIREMENTS



================================================================================
SCENARIO 2
================================================================================

DESCRIPTION:
The ego vehicle is driving on a straight road; the adversarial pedestrian stands behind a bus stop on the right front, then suddenly sprints out onto the road in front of the ego vehicle and stops.

SCENIC CODE:
# BEGIN TOWN
Town = 'Town05'
# END TOWN
# BEGIN IMPORTS
param map = localPath(f'../../assets/maps/CARLA/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
# END IMPORTS
# BEGIN BEHAVIOR
behavior AdvBehavior():
    do CrossingBehavior(ego, globalParameters.OPT_ADV_SPEED, globalParameters.OPT_ADV_DISTANCE) until self in ego.lane or self in ego.lane.successor
    while True:
        take SetWalkingSpeedAction(0)

param OPT_ADV_SPEED = Range(0, 5)
param OPT_ADV_DISTANCE = Range(0, 15)
param OPT_STOP_DISTANCE = Range(0, 1)
# END BEHAVIOR
# BEGIN GEOMETRY
lane = Uniform(*network.lanes)
EgoTrajectory = lane.centerline
EgoSpawnPt = new OrientedPoint on lane.centerline

ego = new Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# END GEOMETRY
# BEGIN SPAWN
param OPT_GEO_BLOCKER_X_DISTANCE = Range(2, 8)
param OPT_GEO_BLOCKER_Y_DISTANCE = Range(15, 50)
param OPT_GEO_X_DISTANCE = Range(-2, 2)
param OPT_GEO_Y_DISTANCE = Range(2, 6)

IntSpawnPt = new OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_BLOCKER_Y_DISTANCE
Blocker = new BusStop right of IntSpawnPt by globalParameters.OPT_GEO_BLOCKER_X_DISTANCE,
    with heading IntSpawnPt.heading,
    with regionContainedIn None
    
SHIFT = globalParameters.OPT_GEO_X_DISTANCE @ globalParameters.OPT_GEO_Y_DISTANCE
pedestrian = new Pedestrian at Blocker offset along IntSpawnPt.heading by SHIFT,
    with heading IntSpawnPt.heading + 90 deg,
    with regionContainedIn None,
    with behavior AdvBehavior()
# END SPAWN
# BEGIN REQUIREMENTS
require distance from ego to pedestrian > 0
require eventually pedestrian in network.drivableRegion
require pedestrian in network.walkableRegion
require eventually ego.canSee(pedestrian, occludingObjects=tuple([])) # Make sure to replace the empty list with a list of all other simulation agents
# END REQUIREMENTS



================================================================================
SCENARIO 3
================================================================================

DESCRIPTION:
The ego vehicle is driving on a straight road; the adversarial pedestrian is hidden behind a vending machine on the right front, and abruptly dashes out onto the road, and stops directly in the path of the ego.

SCENIC CODE:
# BEGIN TOWN
Town = 'Town05'
# END TOWN
# BEGIN IMPORTS
param map = localPath(f'../../assets/maps/CARLA/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
# END IMPORTS
# BEGIN BEHAVIOR
behavior AdvBehavior():
    do CrossingBehavior(ego, globalParameters.OPT_ADV_SPEED, globalParameters.OPT_ADV_DISTANCE) until self in ego.lane or self in ego.lane.successor
    while True:
        take SetWalkingSpeedAction(0)

param OPT_ADV_SPEED = Range(0, 5)
param OPT_ADV_DISTANCE = Range(0, 15)
param OPT_STOP_DISTANCE = Range(0, 1)
# END BEHAVIOR
# BEGIN GEOMETRY
lane = Uniform(*network.lanes)
EgoTrajectory = lane.centerline
EgoSpawnPt = new OrientedPoint on lane.centerline

ego = new Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# END GEOMETRY
# BEGIN SPAWN
param OPT_GEO_BLOCKER_X_DISTANCE = Range(2, 8)
param OPT_GEO_BLOCKER_Y_DISTANCE = Range(15, 50)
param OPT_GEO_X_DISTANCE = Range(-2, 2)
param OPT_GEO_Y_DISTANCE = Range(2, 6)

RightFrontSpawnPt = new OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_BLOCKER_Y_DISTANCE
Blocker = new VendingMachine right of RightFrontSpawnPt by globalParameters.OPT_GEO_BLOCKER_X_DISTANCE,
    with heading RightFrontSpawnPt.heading,
    with regionContainedIn None

SHIFT = globalParameters.OPT_GEO_X_DISTANCE @ globalParameters.OPT_GEO_Y_DISTANCE
pedestrian = new Pedestrian at Blocker offset along RightFrontSpawnPt.heading by SHIFT,
    with heading RightFrontSpawnPt.heading + 90 deg,
    with regionContainedIn None,
    with behavior AdvBehavior()
# END SPAWN
# BEGIN REQUIREMENTS
require not ego.canSee(pedestrian, occludingObjects=tuple([Blocker]))
require eventually pedestrian in network.drivableRegion
require pedestrian in network.walkableRegion
require eventually distance from ego to pedestrian < 5
# END REQUIREMENTS



================================================================================
SCENARIO 4
================================================================================

DESCRIPTION:
The ego vehicle is driving on a straight road; the adversarial pedestrian appears from a driveway on the left and suddenly stop and walk diagonally.

SCENIC CODE:
# BEGIN TOWN
Town = 'Town05'
# END TOWN
# BEGIN IMPORTS
param map = localPath(f'../../assets/maps/CARLA/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
# END IMPORTS
# BEGIN BEHAVIOR
behavior AdvBehavior():
    direction = self.heading + globalParameters.OPT_ADV_DEGREE deg
    take SetWalkingDirectionAction(direction)
    do CrossingBehavior(ego, min_speed=1, threshold=globalParameters.OPT_ADV_DISTANCE, final_speed=globalParameters.OPT_ADV_SPEED)

param OPT_ADV_SPEED = Range(1, 3)
param OPT_ADV_DISTANCE = Range(5, 10)
param OPT_ADV_DEGREE = Range(30, 60)
param OPT_STOP_DURATION = Range(1, 3)
# END BEHAVIOR
# BEGIN GEOMETRY
lane = Uniform(*network.lanes)
EgoTrajectory = lane.centerline
EgoSpawnPt = new OrientedPoint on lane.centerline

ego = new Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# END GEOMETRY
# BEGIN SPAWN
# Parameters for scenario elements
param OPT_GEO_DRIVEWAY_X_DISTANCE = Range(-10, -5)  # Negative range for left side
param OPT_GEO_DRIVEWAY_Y_DISTANCE = Range(5, 15)

# Setting up the spawn point for the adversarial pedestrian
DrivewaySpawnPt = new OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_DRIVEWAY_Y_DISTANCE

# Setup for the adversarial pedestrian appearing from the driveway
SHIFT = Vector(globalParameters.OPT_GEO_DRIVEWAY_X_DISTANCE, 0)
AdvAgent = new Pedestrian at DrivewaySpawnPt offset along DrivewaySpawnPt.heading by SHIFT,
    with heading DrivewaySpawnPt.heading - 90 deg,  # Adjusted for appearing from the left
    with regionContainedIn None,
    with behavior AdvBehavior()
# END SPAWN
# BEGIN REQUIREMENTS
require eventually AdvAgent in network.drivableRegion
require distance from ego to AdvAgent > 0
# END REQUIREMENTS



================================================================================
SCENARIO 5
================================================================================

DESCRIPTION:
The ego vehicle is driving on a straight road; the adversarial pedestrian suddenly appears from behind a parked car on the right front and suddenly stop.

SCENIC CODE:
# BEGIN TOWN
Town = 'Town05'
# END TOWN
# BEGIN IMPORTS
param map = localPath(f'../../assets/maps/CARLA/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
# END IMPORTS
# BEGIN BEHAVIOR
behavior AdvBehavior():
    do CrossingBehavior(ego, globalParameters.OPT_ADV_SPEED, globalParameters.OPT_ADV_DISTANCE) until self in ego.lane or self in ego.lane.successor
    while True:
        take SetWalkingSpeedAction(0)

param OPT_ADV_SPEED = Range(1, 3)
param OPT_ADV_DISTANCE = Range(5, 10)
# END BEHAVIOR
# BEGIN GEOMETRY
lane = Uniform(*network.lanes)
EgoTrajectory = lane.centerline
EgoSpawnPt = new OrientedPoint on lane.centerline

ego = new Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# END GEOMETRY
# BEGIN SPAWN
param OPT_GEO_BLOCKER_X_DISTANCE = Range(2, 8)
param OPT_GEO_BLOCKER_Y_DISTANCE = Range(15, 50)
param OPT_GEO_X_DISTANCE = Range(-2, 2)
param OPT_GEO_Y_DISTANCE = Range(2, 6)

IntSpawnPt = new OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_BLOCKER_Y_DISTANCE
Blocker = new Car right of IntSpawnPt by globalParameters.OPT_GEO_BLOCKER_X_DISTANCE,
    with heading IntSpawnPt.heading,
    with regionContainedIn None

SHIFT = globalParameters.OPT_GEO_X_DISTANCE @ globalParameters.OPT_GEO_Y_DISTANCE
AdvAgent = new Pedestrian at Blocker offset along IntSpawnPt.heading by SHIFT,
    with heading IntSpawnPt.heading + 90 deg,
    with regionContainedIn None,
    with behavior AdvBehavior()
# END SPAWN
# BEGIN REQUIREMENTS
require distance from ego to AdvAgent > 0
require eventually AdvAgent in network.drivableRegion
require AdvAgent.canSee(ego) and not AdvAgent.canSee(ego, occludingObjects=tuple([Blocker]))
require eventually distance from ego to Blocker > 0
# END REQUIREMENTS



================================================================================
SCENARIO 6
================================================================================

DESCRIPTION:
The ego vehicle is turning left at an intersection; the adversarial motorcyclist on the right front pretends to cross the road but brakes abruptly at the edge of the road, causing confusion.

SCENIC CODE:
# BEGIN TOWN
Town = 'Town05'
# END TOWN
# BEGIN IMPORTS
param map = localPath(f'../../assets/maps/CARLA/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
# END IMPORTS
# BEGIN BEHAVIOR
behavior AdvBehavior():
    do CrossingBehavior(ego, globalParameters.OPT_ADV_SPEED, globalParameters.OPT_ADV_DISTANCE) until (distance from self to concatenateCenterlines([traj.centerline for traj in EgoTrajectory])) < globalParameters.OPT_BRAKE_DISTANCE
    while True:
        take SetBrakeAction(globalParameters.OPT_BRAKE)

param OPT_ADV_SPEED = Range(0, 10)
param OPT_ADV_DISTANCE = Range(0, 15)
param OPT_BRAKE_DISTANCE = Range(0, 1)
param OPT_BRAKE = Range(0, 1)
# END BEHAVIOR
# BEGIN GEOMETRY
intersection = Uniform(*filter(lambda i: i.is4Way or i.is3Way, network.intersections))
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.LEFT_TURN, intersection.maneuvers))
egoInitLane = egoManeuver.startLane
EgoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
EgoSpawnPt = new OrientedPoint in egoInitLane.centerline

ego = new Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# END GEOMETRY
# BEGIN SPAWN
# Define the spawn point for the adversarial motorcyclist on the right front of the ego vehicle at the edge of the road
param OPT_GEO_Y_DISTANCE = Range(20, 30)
param OPT_GEO_X_DISTANCE = Range(1, 3)

# Project a point in front of the ego vehicle
projectPt = new OrientedPoint following roadDirection from ego for globalParameters.OPT_GEO_Y_DISTANCE

# Define the spawn point for the adversarial motorcyclist at the edge of the road
AdvSpawnPoint = new OrientedPoint right of projectPt by globalParameters.OPT_GEO_X_DISTANCE

# Spawn the adversarial motorcyclist
AdvAgent = new Motorcycle at AdvSpawnPoint,
    with regionContainedIn None,
    with behavior AdvBehavior()
# END SPAWN
# BEGIN REQUIREMENTS
require distance from ego to intersection > 0
require eventually distance from ego to AdvAgent < 10
require eventually AdvAgent in network.drivableRegion
require eventually ego in network.drivableRegion
require 70 deg <= RelativeHeading(AdvAgent, ego) <= 110 deg
# END REQUIREMENTS



================================================================================
SCENARIO 7
================================================================================

DESCRIPTION:
The ego vehicle is turning left at an intersection; the adversarial pedestrian on the opposite sidewalk suddenly crosses the road from the right front and stops in the middle of the intersection.

SCENIC CODE:
# BEGIN TOWN
Town = 'Town05'
# END TOWN
# BEGIN IMPORTS
param map = localPath(f'../../assets/maps/CARLA/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
# END IMPORTS
# BEGIN BEHAVIOR
behavior AdvBehavior():
    do CrossingBehavior(ego, globalParameters.OPT_ADV_SPEED, globalParameters.OPT_ADV_DISTANCE) until self in ego.lane or self in ego.lane.successor
    while not self.inIntersection:
        wait
    take SetWalkingSpeedAction(0)

param OPT_ADV_SPEED = Range(0, 5)
param OPT_ADV_DISTANCE = Range(0, 15)
# END BEHAVIOR
# BEGIN GEOMETRY
intersection = Uniform(*filter(lambda i: i.is4Way or i.is3Way, network.intersections))
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.LEFT_TURN, intersection.maneuvers))
egoInitLane = egoManeuver.startLane
EgoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
EgoSpawnPt = new OrientedPoint in egoInitLane.centerline

ego = new Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# END GEOMETRY
# BEGIN SPAWN
param OPT_GEO_X_DISTANCE = Range(2, 8)
param OPT_GEO_Y_DISTANCE = Range(-5, 5)

# Define the spawn point for the adversarial pedestrian
SidewalkSpawnPt = new OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE
SHIFT = globalParameters.OPT_GEO_X_DISTANCE @ globalParameters.OPT_GEO_Y_DISTANCE

# Place the adversarial pedestrian on the opposite sidewalk
pedestrian = new Pedestrian at SidewalkSpawnPt offset along SidewalkSpawnPt.heading by SHIFT,
    with heading SidewalkSpawnPt.heading + 90 deg,
    with regionContainedIn None,
    with behavior AdvBehavior()
# END SPAWN
# BEGIN REQUIREMENTS
require distance from ego to intersection < 15
require eventually pedestrian in network.drivableRegion
require eventually ego in intersection implies pedestrian in intersection
# END REQUIREMENTS



================================================================================
SCENARIO 8
================================================================================

DESCRIPTION:
The ego vehicle is turning right at an intersection; the adversarial motorcyclist on the opposite sidewalk abruptly crosses the road from the right front and comes to a halt in the center of the intersection.

SCENIC CODE:
# BEGIN TOWN
Town = 'Town05'
# END TOWN
# BEGIN IMPORTS
param map = localPath(f'../../assets/maps/CARLA/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
# END IMPORTS
# BEGIN BEHAVIOR
behavior AdvBehavior():
    do CrossingBehavior(ego, globalParameters.OPT_ADV_SPEED, globalParameters.OPT_ADV_DISTANCE) until (distance from self to concatenateCenterlines([traj.centerline for traj in EgoTrajectory])) < globalParameters.OPT_STOP_DISTANCE
    while True:
        take SetSpeedAction(0)

param OPT_ADV_SPEED = Range(0, 10)
param OPT_ADV_DISTANCE = Range(0, 15)
param OPT_STOP_DISTANCE = Range(0, 1)
# END BEHAVIOR
# BEGIN GEOMETRY
intersection = Uniform(*filter(lambda i: i.is4Way or i.is3Way, network.intersections))
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.RIGHT_TURN, intersection.maneuvers))
egoInitLane = egoManeuver.startLane
EgoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
EgoSpawnPt = new OrientedPoint in egoInitLane.centerline

ego = new Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# END GEOMETRY
# BEGIN SPAWN
# Parameters for scenario elements
param OPT_GEO_X_DISTANCE = Range(5, 10)
param OPT_GEO_Y_DISTANCE = Range(2, 5)

# Setup for the adversarial motorcyclist starting on the opposite sidewalk
IntSpawnPt = new OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE

# Assuming the sidewalk is already defined in the geometry part.
AdvAgent = new Motorcycle at IntSpawnPt offset by (globalParameters.OPT_GEO_X_DISTANCE @ 0),
    with heading IntSpawnPt.heading + 90 deg,  # Perpendicular to the road, crossing from the right front of the ego vehicle
    with regionContainedIn None,
    with behavior AdvBehavior()
# END SPAWN
# BEGIN REQUIREMENTS
require distance from ego to intersection > 0
require eventually ego in intersection implies AdvAgent in intersection
require eventually AdvAgent in network.drivableRegion
require AdvAgent in network.walkableRegion
# END REQUIREMENTS



================================================================================
SCENARIO 9
================================================================================

DESCRIPTION:
The ego vehicle is turning right at an intersection; the adversarial pedestrian on the left front suddenly crosses the road and stops in the middle of the intersection, blocking the ego vehicle's path.

SCENIC CODE:
# BEGIN TOWN
Town = 'Town05'
# END TOWN
# BEGIN IMPORTS
param map = localPath(f'../../assets/maps/CARLA/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
# END IMPORTS
# BEGIN BEHAVIOR
behavior AdvBehavior():
    do CrossingBehavior(ego, globalParameters.OPT_ADV_SPEED, globalParameters.OPT_ADV_DISTANCE) until self in ego.lane or self in ego.lane.successor
    while True:
        if self in intersection:
            take SetWalkingSpeedAction(0)

param OPT_ADV_SPEED = Range(0, 5)
param OPT_ADV_DISTANCE = Range(0, 15)
param OPT_STOP_DISTANCE = Range(0, 1)
# END BEHAVIOR
# BEGIN GEOMETRY
intersection = Uniform(*filter(lambda i: i.is4Way or i.is3Way, network.intersections))
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.RIGHT_TURN, intersection.maneuvers))
egoInitLane = egoManeuver.startLane
EgoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
EgoSpawnPt = new OrientedPoint in egoInitLane.centerline

ego = new Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# END GEOMETRY
# BEGIN SPAWN
param OPT_GEO_X_DISTANCE = Range(-8, -2)  # Adjusted for left side
param OPT_GEO_Y_DISTANCE = Range(0, 10)

IntSpawnPt = new OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE
SHIFT = globalParameters.OPT_GEO_X_DISTANCE @ globalParameters.OPT_GEO_Y_DISTANCE
pedestrian = new Pedestrian at IntSpawnPt offset along IntSpawnPt.heading by SHIFT,
    with heading IntSpawnPt.heading - 90 deg,  # Adjusted for coming from the left
    with regionContainedIn None,
    with behavior AdvBehavior()
# END SPAWN
# BEGIN REQUIREMENTS
require distance from ego to intersection > 0
require eventually distance from ego to pedestrian < 5
require eventually pedestrian in network.drivableRegion
require pedestrian in network.walkableRegion
require eventually 70 deg <= RelativeHeading(pedestrian, ego) <= 110 deg
# END REQUIREMENTS



================================================================================
SCENARIO 10
================================================================================

DESCRIPTION:
The ego vehicle is turning left at an intersection; the adversarial cyclist on the left front suddenly stops in the middle of the intersection and dismounts, obstructing the ego vehicle's path.

SCENIC CODE:
# BEGIN TOWN
Town = 'Town05'
# END TOWN
# BEGIN IMPORTS
param map = localPath(f'../../assets/maps/CARLA/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
# END IMPORTS
# BEGIN BEHAVIOR
behavior AdvBehavior():
    do CrossingBehavior(ego, globalParameters.OPT_ADV_SPEED, globalParameters.OPT_ADV_DISTANCE) until (distance from self to concatenateCenterlines([traj.centerline for traj in EgoTrajectory])) < globalParameters.OPT_STOP_DISTANCE
    while True:
        take SetSpeedAction(0)

param OPT_ADV_SPEED = Range(0, 10)
param OPT_ADV_DISTANCE = Range(0, 15)
param OPT_STOP_DISTANCE = Range(0, 1)
# END BEHAVIOR
# BEGIN GEOMETRY
intersection = Uniform(*filter(lambda i: i.is4Way or i.is3Way, network.intersections))
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.LEFT_TURN, intersection.maneuvers))
egoInitLane = egoManeuver.startLane
EgoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
EgoSpawnPt = new OrientedPoint in egoInitLane.centerline

ego = new Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# END GEOMETRY
# BEGIN SPAWN
# Defining the adversarial cyclist's position relative to the ego vehicle
IntSpawnPt = new OrientedPoint at ego offset along ego.heading by Vector(5, 0)

# Setting up the adversarial cyclist
AdvAgent = new Bicycle left of IntSpawnPt by 2,
    with heading IntSpawnPt.heading,
    with regionContainedIn None,
    with behavior AdvBehavior()
# END SPAWN
# BEGIN REQUIREMENTS
require eventually ego in intersection implies AdvAgent in intersection
require distance from ego to intersection < 15
require eventually distance from ego to AdvAgent < 5
require -110 deg <= RelativeHeading(AdvAgent, ego) <= -70 deg # not sure abt this one 
# END REQUIREMENTS



================================================================================
SCENARIO 12
================================================================================

DESCRIPTION:
The ego vehicle is changing to the right lane; the adversarial car is driving parallel to the ego and blocking its path.

SCENIC CODE:
# BEGIN TOWN
Town = 'Town05'
# END TOWN
# BEGIN IMPORTS
param map = localPath(f'../scenic_scenarios/assets/maps/CARLA/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
# END IMPORTS
# BEGIN BEHAVIOR
behavior AdvBehavior():
    while True:
        take SetVelocityAction(*ego.velocity)
        self.heading = ego.heading  # Keeps the adversarial vehicle heading aligned with the ego
# END BEHAVIOR
# BEGIN GEOMETRY
# Identifying lane sections where the ego vehicle can change to the right lane
laneSecsWithRightLaneChange = []
for lane in network.lanes:
    for laneSec in lane.sections:
        if laneSec._laneToRight is not None and laneSec._laneToRight.isForward == laneSec.isForward:
            laneSecsWithRightLaneChange.append(laneSec)

# Selecting a random lane section from identified sections for the ego vehicle
EgoLaneSec = Uniform(*laneSecsWithRightLaneChange)
EgoSpawnPt = new OrientedPoint in EgoLaneSec.centerline

# Placing the ego vehicle in the lane section and specifying it is changing to the right lane
ego = new Car at EgoSpawnPt
# END GEOMETRY
# BEGIN SPAWN
# Parameters for scenario elements
param OPT_GEO_X_DISTANCE = Range(0, 3)
# Setup for the adversarial car parallel to the ego vehicle
AdvSpawnPoint = new OrientedPoint right of ego by globalParameters.OPT_GEO_X_DISTANCE
AdvAgent = new Car at AdvSpawnPoint,
    with heading ego.heading,  # The agent is parallel to the ego vehicle
    with behavior AdvBehavior()
# END SPAWN
# BEGIN REQUIREMENTS
require distance from ego to AdvAgent < 5
require eventually ego.road == AdvAgent.road
# END REQUIREMENTS



================================================================================
SCENARIO 16
================================================================================

DESCRIPTION:
The ego approaches a parked car that is blocking its lane and must use the opposite lane to bypass the vehicle, cautiously monitoring oncoming traffic, and suddenly encounters a jaywalking pedestrian, requiring the ego to quickly assess the situation and respond appropriately to avoid a collision.

SCENIC CODE:
# BEGIN TOWN
Town = 'Town01'
# END TOWN
# BEGIN IMPORTS
param map = localPath(f'../scenic_scenarios/assets/maps/CARLA/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
# END IMPORTS
# BEGIN BEHAVIOR
behavior AdvBehavior():
    do CrossingBehavior(ego, globalParameters.OPT_ADV_SPEED, globalParameters.OPT_ADV_DISTANCE) until self in ego.lane or self in ego.lane.successor

param OPT_ADV_SPEED = Range(0, 5)
param OPT_ADV_DISTANCE = Range(0, 15)
# END BEHAVIOR
# BEGIN GEOMETRY
# Collecting lane sections that have a left lane (opposite traffic direction) and no right lane (single forward road)
laneSecsWithLeftLane = []
for lane in network.lanes:
    for laneSec in lane.sections:
        if laneSec._laneToLeft is not None and laneSec._laneToRight is None:
            if laneSec._laneToLeft.isForward != laneSec.isForward:
                laneSecsWithLeftLane.append(laneSec)

# Selecting a random lane section that matches the criteria
EgoLaneSec = Uniform(*laneSecsWithLeftLane)
EgoSpawnPt = new OrientedPoint in EgoLaneSec.centerline
# END GEOMETRY
# BEGIN SPAWN
# Parameters for scenario elements
param OPT_GEO_X_DISTANCE = Range(2, 4)
param OPT_GEO_Y_DISTANCE = Range(8, 15)

# Setup for the pedestrian who enters the road from the side
IntSpawnPt = new OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE
SHIFT = globalParameters.OPT_GEO_X_DISTANCE @ globalParameters.OPT_GEO_Y_DISTANCE
AdvAgent = new Pedestrian at IntSpawnPt offset along IntSpawnPt.heading by SHIFT,
    with heading IntSpawnPt.heading + 90 deg,  # Perpendicular to the road, crossing the street
    with regionContainedIn None,
    with behavior AdvBehavior()

ego = new Car at EgoSpawnPt, with blueprint EGO_MODEL
# END SPAWN
# BEGIN MISC
parkedCar = new Car at EgoSpawnPt offset along EgoSpawnPt.heading by 1 @ 9, 
    with blueprint "vehicle.mercedes.sprinter", 
    with regionContainedIn None
# END MISC
# BEGIN REQUIREMENTS
require distance from ego to parkedCar < 10
require eventually ego.canSee(AdvAgent, occludingObjects=tuple([])) # Make sure to replace the empty list with a list of all other simulation agents
require AdvAgent in network.walkableRegion
require eventually AdvAgent in network.drivableRegion
# END REQUIREMENTS



================================================================================
SCENARIO 21
================================================================================

DESCRIPTION:
The ego is driving straight through an intersection when a crossing vehicle runs the red light and unexpectedly accelerates, forcing the ego to quickly reassess the situation and perform a collision avoidance maneuver.

SCENIC CODE:
# BEGIN TOWN
Town = 'Town05'
# END TOWN
# BEGIN IMPORTS
param map = localPath(f'../scenic_scenarios/assets/maps/CARLA/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
# END IMPORTS
# BEGIN BEHAVIOR
param OPT_ADV_SPEED = Range(5, 10)
param OPT_THROTTLE = Range(0.5, 1)
behavior AdvBehavior():
    do FollowLaneBehavior(target_speed=globalParameters.OPT_ADV_SPEED) until withinDistanceToRedYellowTrafficLight(self, 10)
    take SetThrottleAction(globalParameters.OPT_THROTTLE)
# END BEHAVIOR
# BEGIN GEOMETRY
intersection = Uniform(*network.intersections)
egoInitLane = Uniform(*intersection.incomingLanes)
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT, egoInitLane.maneuvers))
EgoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
EgoSpawnPt = new OrientedPoint in egoManeuver.startLane.centerline

# Setting up the ego vehicle at the initial position
ego = new Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# END GEOMETRY
# BEGIN SPAWN
AdvManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT, egoManeuver.conflictingManeuvers))
AdvAgent = new Car in AdvManeuver.startLane.centerline,
    with regionContainedIn None,
    with behavior AdvBehavior()
# END SPAWN
# BEGIN REQUIREMENTS
require eventually ego in intersection implies AdvAgent in intersection
require distance from ego to intersection < 15
require eventually distance from ego to AdvAgent < 10
require -110 deg <= RelativeHeading(AdvAgent, ego) <= -70 deg
# END REQUIREMENTS



================================================================================
SCENARIO 26
================================================================================

DESCRIPTION:
The ego starts an unprotected left turn at an intersection while yielding to an oncoming car when the oncoming car's throttle malfunctions, leading to an unexpected acceleration and forcing the ego to quickly modify its turning path to avoid a collision.

SCENIC CODE:
# BEGIN TOWN
Town = 'Town05'
# END TOWN
# BEGIN IMPORTS
param map = localPath(f'../scenic_scenarios/assets/maps/CARLA/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
# END IMPORTS
# BEGIN BEHAVIOR
param OPT_MALFUNCTION_THROTTLE = Range(0.8, 1)  # High throttle intensity due to malfunction

behavior AdvBehavior():
    while True:
        take SetThrottleAction(globalParameters.OPT_MALFUNCTION_THROTTLE)
        wait
# END BEHAVIOR
# BEGIN GEOMETRY

## MONITORS
monitor TrafficLights():
    freezeTrafficLights()
    while True:
        if withinDistanceToTrafficLight(ego, 100):
            setClosestTrafficLightStatus(ego, "green")
        if withinDistanceToTrafficLight(AdvAgent, 100):
            setClosestTrafficLightStatus(AdvAgent, "green")
        wait

intersection = Uniform(*filter(lambda i: i.is4Way and i.isSignalized, network.intersections))
egoInitLane = Uniform(*intersection.incomingLanes)
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.LEFT_TURN, egoInitLane.maneuvers))
EgoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
EgoSpawnPt = new OrientedPoint in egoManeuver.startLane.centerline

# Setting up the ego vehicle at the initial position
ego = new Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# END GEOMETRY
# BEGIN SPAWN
# Parameters for scenario elements
param OPT_GEO_X_DISTANCE = Range(-8, 0)  # Offset for the agent in the opposite lane
param OPT_GEO_Y_DISTANCE = Range(10, 30)

# Setup for the adversarial car coming from the opposite direction
SHIFT = globalParameters.OPT_GEO_X_DISTANCE @ globalParameters.OPT_GEO_Y_DISTANCE
IntSpawnPt = new OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE

AdvAgent = new Car at IntSpawnPt offset along IntSpawnPt.heading by SHIFT,
    with heading IntSpawnPt.heading + 180 deg,  # The agent is facing the opposite direction, indicating oncoming
    with regionContainedIn network.drivableRegion,  # Positioned in the drivable region
    with behavior AdvBehavior()
# END SPAWN
# BEGIN REQUIREMENTS
require distance from ego to intersection < 15
require eventually ego in intersection implies AdvAgent in intersection
require eventually distance from ego to AdvAgent < 10
# END REQUIREMENTS



================================================================================
SCENARIO 31
================================================================================

DESCRIPTION:
The ego is performing a right turn at an intersection when the crossing car suddenly speeds up, entering the intersection and causing the ego to brake abruptly to avoid a collision.

SCENIC CODE:
# BEGIN TOWN
Town = 'Town05'
# END TOWN
# BEGIN IMPORTS
param map = localPath(f'../scenic_scenarios/assets/maps/CARLA/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
# END IMPORTS
# BEGIN BEHAVIOR
behavior AdvBehavior():
    while (distance to self) > 60:
        wait  # The adversarial vehicle waits until it is close enough to affect the ego's maneuver.
    do FollowTrajectoryBehavior(globalParameters.OPT_ADV_SPEED, advTrajectory) until (distance from self to concatenateCenterlines([traj.centerline for traj in EgoTrajectory])) < globalParameters.OPT_ACCEL_DISTANCE
    # Accelerates suddenly as it approaches the intersection.
    while True:
        take SetThrottleAction(globalParameters.OPT_THROTTLE)  # Applies throttle to increase speed rapidly.

param OPT_ADV_SPEED = Range(5, 15)  # The speed at which the adversarial vehicle approaches the intersection.
param OPT_ACCEL_DISTANCE = Range(0, 4)  # The distance at which the adversarial vehicle starts its sudden acceleration.
param OPT_THROTTLE = Range(0.5, 1.0)  # The intensity of the throttle during the acceleration.
# END BEHAVIOR
# BEGIN GEOMETRY
intersection = Uniform(*filter(lambda i: i.is4Way or i.is3Way, network.intersections))
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.RIGHT_TURN, intersection.maneuvers))
egoInitLane = egoManeuver.startLane
EgoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
EgoSpawnPt = new OrientedPoint in egoInitLane.centerline

ego = new Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# END GEOMETRY
# BEGIN SPAWN
# Defining adversarial maneuvers as those conflicting with the ego's straight path
advManeuvers = filter(lambda i: i.type == ManeuverType.STRAIGHT, egoManeuver.conflictingManeuvers)
advManeuver = Uniform(*advManeuvers)
advTrajectory = [advManeuver.startLane, advManeuver.connectingLane, advManeuver.endLane]
IntSpawnPt = new OrientedPoint in advManeuver.startLane.centerline

# Setting up the adversarial agent
AdvAgent = new Car at IntSpawnPt,
    with heading IntSpawnPt.heading,
    with regionContainedIn None,
    with behavior AdvBehavior()
# END SPAWN
# BEGIN REQUIREMENTS
require distance from ego to intersection < 15
require eventually ego in intersection implies AdvAgent in intersection
require -110 deg <= RelativeHeading(AdvAgent, ego) <= -70 deg
# END REQUIREMENTS



================================================================================
SCENARIO 36
================================================================================

DESCRIPTION:
The ego vehicle is approaching the intersection; the adversarial car (on the left) suddenly accelerates and enters the intersection first and suddenly stop.

SCENIC CODE:
# BEGIN TOWN
Town = 'Town05'
# END TOWN
# BEGIN IMPORTS
param map = localPath(f'../../../../../assets/maps/CARLA/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
# END IMPORTS
# BEGIN BEHAVIOR
behavior AdvBehavior():
    while (distance to self) > 60:
        wait  # Wait until the vehicle is close enough to influence the ego's path.
    # Accelerates towards the intersection.
    do FollowLaneBehavior(globalParameters.OPT_ADV_SPEED) until self in intersection
    # Once in the intersection, it stops suddenly.
    while True:
        take SetSpeedAction(0)  # Initiates a sudden stop.

param OPT_ADV_SPEED = Range(10, 20)  # Speed at which the vehicle accelerates towards the intersection.
param OPT_INTERSECTION_DISTANCE = Range(0, 5)  # The critical distance to the intersection to start stopping.
# END BEHAVIOR
# BEGIN GEOMETRY
intersection = Uniform(*filter(lambda i: i.is4Way or i.is3Way, network.intersections))
egoApproachLane = Uniform(*intersection.incomingLanes)
EgoSpawnPt = new OrientedPoint in egoApproachLane.centerline

ego = new Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# END GEOMETRY
# Setup the adversarial agent's spawn point to the left of the ego vehicle
param OPT_GEO_Y_DISTANCE = Range(0, 30)
advLane = ego.laneSection.laneToLeft
IntSpawnPt = new OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE
distance = advLane.centerline.start.distanceTo(egoApproachLane.centerline.start) # finding the distance between the two lanes
projectPt = new OrientedPoint left of IntSpawnPt by distance

# Spawn the Adversarial Agent as a generic object placeholder { AdvObject }

AdvAgent = new Car at projectPt,
    with regionContainedIn None,
    with behavior AdvBehavior()"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=message)
            ]
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
    )

    accumulated_contents = ""

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text is not None:
            accumulated_contents += chunk.text

    return accumulated_contents
    print(chunk.text, end="")


for scenario_folder in scenario_folders:
    scenic_files = []
    for root, dirs, files in os.walk(os.path.join(SCENARIOS_FOLDER, scenario_folder)):
        for file in files:
            if file.endswith('.scenic'):
                scenic_files.append(os.path.join(root, file))

    scenario_number = re.match(r'scenario_(\d+)', scenario_folder).group(1)
    scenario_desc = list_of_scenarios[int(scenario_number)-1]
    # print(f"Evaluating scenario {scenario_number}: {scenario_desc}")

    # randomly select 10 scenic files without replacement
    selected_scenic_files = random.sample(
        scenic_files, min(10, len(scenic_files)))
    for scenic_file in selected_scenic_files:
        with open(scenic_file, 'r') as file:
            scenic_script = file.read()
            scenic_script = '\n'.join(scenic_script.splitlines()[1:])
        # format the eval prompt
        formatted_eval_prompt = eval_prompt.format(
            desc=scenario_desc,
            curr_script=scenic_script
        )

        # print(
        #    f"Formatted eval prompt for {scenic_file}:\n{formatted_eval_prompt}")
        # genai call to evaluate the scenario
        response = generate(formatted_eval_prompt)
        # print("\n\n" + str(response) + "#" * 20 + "\n\n")
        # save the response to a file
        output_file = f"eval_results/gemini/{scenario_folder}_{os.path.basename(scenic_file)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as file:
            file.write(response)
