# Assignment Part 02 - Discussion for Improving Current Model

## 1. Possible Continuous Inputs
Currently, the `PyRace2D.observe()` function converts the continuous pixel distances given by the radars into discrete values between `0` and `10` via:
```python
ret[i] = int(r[1] / 20)
```
Moreover, the existing Q-Table approach maps these already discrete bounds into bucketized dimensions.
**Improvement:** For a Deep Q-Network or any Policy Gradient algorithm, the state-to-bucket process is completely redundant. Neural networks natively digest continuous variables effectively and generalize much better across ranges of floating point values. By avoiding the `int(r[1] / 20)` rounding limit, we would supply the agent with a significantly finer grain of distance measures (e.g., precise to `1` pixel), avoiding quantization errors (where distances of 25 and 39 are both rounded to 1 despite being importantly functionally different depending on speed) and thus yielding a smoother, better-optimized driving policy.

## 2. Possible Range of Actions
Presently, the agent uses three absolute choices at every frame.
```python
if action == 0: self.car.speed += 2
elif action == 1: self.car.angle += 5
elif action == 2: self.car.angle -= 5
```
**Improvement:**
1. **Explicit Braking:** We should consider an explicit `BRAKE` action. Currently the car decelerates implicitly across all frames (`self.speed -= 0.5`). Without an explicit brake, the car might not be able to avoid collisions in fast turns. If we add `elif action == 3: self.car.speed -= 3`, the agent can selectively reduce speed when approaching tight turns as detected by short distant front-facing radar.
2. **Combination of Action:** Separating acceleration from steering (e.g. MultiDiscrete environment) would be highly useful. Allowing the car to steer and accelerate/brake *simultaneously* mimics actual racecar controls, unlike the present layout where applying speed makes the vehicle forego any turning input for that frame. Continuous Actions instead of discrete ones could also be adopted if we transition to Deep Deterministic Policy Gradients (DDPG) or Soft Actor Critic (SAC) models in the future.

## 3. New Engineered Reward Function
Existing implementation checks:
```python
if not self.car.is_alive:
    reward = -10000 + self.car.distance
elif self.car.goal:
    reward = 10000
# evaluate() is not passing continuous reward by default, it relies on distance logic only occasionally triggering check flags.
```
**Improvement:** 
We can design an auxiliary dense reward function replacing or augmenting the sparse large values:
*   **Progress Tracking:** Reward proportional to the change in distance to the track centerline. Keeping the center leads to optimal lap times.
*   **Velocity Vector Penalty:** Instead of rewarding straight distance, which might mean just accelerating blindly into walls before death, we should reward velocity vector matching the track's directional centerline vector.
*   **Safety Penalty:** Give small negative rewards or scale velocity reward downwards whenever side-radar distances become extremely low. This teaches the agent to proactively stabilize rather than reacting only after reaching `-10000` from death.
*   **Control Effort Penalty:** Providing a small but consistent penalty for zigzagging (`action == 1` then immediately `action == 2`). This stops jittery behavior by promoting smooth steering.
