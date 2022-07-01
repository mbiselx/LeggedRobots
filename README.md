# LeggedRobots
## Miniprojects for the [MICRO-507 : Legged Robots](https://edu.epfl.ch/coursebook/en/legged-robots-MICRO-507) course 


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project 1](#about-the-project-1)
* [About the Project 2](#about-the-project-2)
* [Folder Structure](#folder-structure)
* [Videos](#videos)
* [Contact](#contacts)

<!-- ABOUT THE PROJECT -->
## About The Project 1
Project 1 is about planning the Center of Mass (CoM) trajectory for a (simulated) biped Atlas robot. The technique used is the Divergent Component of Motion (DCM).
The resulting gaits are analyzed and discussed, and the conclusion is reached that this method requires quite a few simplifications, some of which are not valid in the real world.

## About The Project 2
Project 2 adresses quadruped locomotion. Gaits are generated for (simulated) A1 quadruped, first using Central Pattern Generators (CPG), and the using Deep Reinforcement Learning (DRL)
The resulting gaits are analyzed and discussed, and the conclusion is reached that CPG methods are relatively simple and quick to implement, however, they require a lot of parameter tuning to get them to work nicely. DRL techniques require no (manual) parameter tuning, but instead require the setting an appropriate reward function, which can be difficult to guesstimate. 


<!-- FOLDER Structure -->
## Folder Structure
| Folder Name             | Comment                                                                                            |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| Project_1               | folder containing the project 1                                                                    |
| - code                  | python code for the project                                                                        |
| Project_2               | folder containing the project 2                                                                    |
| - code                  | python code for the project                                                                        |

<!-- VIDEOS -->
## Videos
### Project 1
#### Fast Locomotion
https://user-images.githubusercontent.com/58890541/176934334-e5a097c3-9a74-4e3b-a458-2961b488f5af.mp4

#### Normal Locomotion
https://user-images.githubusercontent.com/58890541/176934359-aefaad11-c606-4492-9c89-2044f74ad744.mp4

### Project 2
#### CPG
##### Pace
https://user-images.githubusercontent.com/58890541/176934022-2641cf2d-0b81-4d27-afb7-af446c180431.mov

##### Walk
https://user-images.githubusercontent.com/58890541/176933999-f80c5961-08d8-4a28-a614-270d01db327e.mov

##### Bound
https://user-images.githubusercontent.com/58890541/176933979-6763f09a-711d-48cc-9938-b843864f70ba.mov

##### Trot
https://user-images.githubusercontent.com/58890541/176933956-f814b95e-d57a-4f14-b5cc-7d4d49f90052.mov

#### DRL
##### PPO with Cartesian PD
https://user-images.githubusercontent.com/58890541/176933792-0b2100dc-b911-4128-a57d-6c3d512e3363.mov

##### PPO without Cartesian PD
https://user-images.githubusercontent.com/58890541/176933813-2b8b16cd-2062-4adf-bad7-53f3287e5cb8.mov

<!-- CONTACT -->
## Contacts
Biselx Michael - michael.biselx@epfl.ch    <br />
Bumann Samuel  - samuel.bumann@epfl.ch     <br />
[lvuilleu](https://github.com/lvuilleu)    <br />


Project Link: [https://github.com/mbiselx/LeggedRobots](https://github.com/mbiselx/LeggedRobots)
