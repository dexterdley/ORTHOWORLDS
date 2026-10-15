"""
multicar_env.py — Two-car CarRacing-v3 (gymnasium) with a scripted opponent.

The ego car is controlled through step(action) exactly like CarRacing-v3. The opponent
(blue) follows the track centreline with a pure-pursuit autopilot by default, or takes
actions from set_opponent_action() when autopilot=False (e.g. a trained PPO policy).

Only the ego car's wheels earn tile rewards; the opponent's wheels still register road
contact so it gets road friction. The two hulls collide with each other (wheels don't).
"""
import math

import numpy as np
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.envs.box2d.car_racing import TRACK_WIDTH, CarRacing, FrictionDetector

OPPONENT_COLOR = (0.1, 0.25, 0.9)


class _EgoOnlyRewardDetector(FrictionDetector):
    """FrictionDetector that tracks road contact for every wheel but only rewards the ego car."""

    def _contact(self, contact, begin):
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        for tile, obj in ((u1, u2), (u2, u1)):
            if tile and "road_friction" in tile.__dict__ and obj and obj.__dict__.get("no_reward"):
                if begin:
                    obj.tiles.add(tile)
                else:
                    obj.tiles.discard(tile)
                return
        super()._contact(contact, begin)


class MultiCarRacing(CarRacing):
    def __init__(self, render_mode="rgb_array", autopilot=True, opponent_speed=1.0, **kwargs):
        super().__init__(render_mode=render_mode, continuous=True, **kwargs)
        self.autopilot = autopilot
        self.opponent_speed = opponent_speed
        self.opponent_car = None
        self.opponent_action = np.zeros(3)
        self.progress = {}  # car name -> continuous track progress in tiles

    # ------------------------------------------------------------------ setup
    def reset(self, *, seed=None, options=None):
        if self.opponent_car is not None:
            self.opponent_car.destroy()
            self.opponent_car = None
        super().reset(seed=seed, options=options)
        self.track_xy = np.array([[t[2], t[3]] for t in self.track])

        # Spawn the opponent two centreline points ahead, offset towards one road edge
        _, beta, x, y = self.track[2]
        lateral = 0.5 * TRACK_WIDTH
        self.opponent_car = Car(self.world, beta, x - lateral * math.cos(beta), y - lateral * math.sin(beta))
        self.opponent_car.hull.color = OPPONENT_COLOR
        for w in self.opponent_car.wheels:
            w.no_reward = True
        self.opponent_action = np.zeros(3)

        # Replace the listener before the opponent's first world step
        self._listener_ref = _EgoOnlyRewardDetector(self, self.lap_complete_percent)
        self.world.contactListener = self._listener_ref

        # Draw the opponent into every top-down frame (state pixels and render())
        ego_draw = self.car.draw

        def draw_both(surface, zoom, translation, angle, draw_particles=True):
            ego_draw(surface, zoom, translation, angle, draw_particles)
            self.opponent_car.draw(surface, zoom, translation, angle, draw_particles)

        self.car.draw = draw_both
        self.progress = {"ego": 0.0, "opponent": 2.0}
        self._nearest = {"ego": 0, "opponent": 2}
        self.state = self._render("state_pixels")
        return self.state, {}

    def set_opponent_action(self, action):
        self.opponent_action = np.asarray(action, dtype=np.float64).flatten()[:3]

    # ------------------------------------------------------------------ step
    def step(self, action):
        if self.opponent_car is not None:
            if self.autopilot:
                self.opponent_action = self.autopilot_action(self.opponent_car)
            steer, gas, brake = self.opponent_action
            self.opponent_car.steer(-steer)
            self.opponent_car.gas(gas)
            self.opponent_car.brake(brake)
            self.opponent_car.step(1.0 / 50.0)
        if action is not None:
            action = np.asarray(action, dtype=np.float64).flatten()[:3]
        out = super().step(action)
        if self.opponent_car is not None:
            self._update_progress("ego", self.car)
            self._update_progress("opponent", self.opponent_car)
        return out

    def _nearest_index(self, car):
        return int(np.argmin(np.sum((self.track_xy - np.asarray(car.hull.position)) ** 2, axis=1)))

    def _update_progress(self, name, car):
        n = len(self.track_xy)
        i = self._nearest_index(car)
        d = (i - self._nearest[name] + n // 2) % n - n // 2  # signed, wrap-aware index delta
        self.progress[name] += d
        self._nearest[name] = i

    def autopilot_action(self, car, lookahead=5):
        """Pure pursuit on the track centreline, slowing for upcoming corners."""
        n = len(self.track_xy)
        i = self._nearest_index(car)
        target = self.track_xy[(i + lookahead) % n]
        lx, ly = car.hull.GetLocalPoint((float(target[0]), float(target[1])))
        heading_err = math.atan2(lx, ly)  # > 0: target is to the right
        steer = float(np.clip(2.0 * heading_err, -1.0, 1.0))

        # Corner sharpness over the next ~12 points sets the target speed
        betas = np.array([self.track[(i + k) % n][1] for k in range(0, 12)])
        turn = float(np.max(np.abs(np.unwrap(betas) - betas[0])))
        target_speed = self.opponent_speed * (75.0 - 45.0 * min(turn / 1.2, 1.0))
        speed = float(np.linalg.norm(car.hull.linearVelocity))
        gas = 0.6 if speed < target_speed else 0.0
        brake = 0.4 if speed > target_speed + 8.0 else 0.0
        return np.array([steer, gas, brake])

    def close(self):
        if self.opponent_car is not None:
            self.opponent_car.destroy()
            self.opponent_car = None
        super().close()
