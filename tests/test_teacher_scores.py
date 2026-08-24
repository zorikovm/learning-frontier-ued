import unittest

import jax.numpy as jnp
import numpy as np

from jaxued.teacher_scores import compute_mna_advantages, solved_mna_score


class TeacherScoreTest(unittest.TestCase):
    def test_mna_is_teacher_only_shape_preserving_signal(self):
        values = jnp.array([[1.0], [1.0], [0.5]])
        rewards = jnp.array([[0.0], [0.0], [1.0]])
        dones = jnp.array([[False], [False], [True]])

        advantages, targets = compute_mna_advantages(
            0.995, 0.98, jnp.array([0.0]), values, rewards, dones
        )

        self.assertEqual(advantages.shape, values.shape)
        self.assertEqual(targets.shape, values.shape)
        np.testing.assert_allclose(np.asarray(targets), np.asarray(advantages + values))
        self.assertTrue(np.all(np.asarray(advantages) <= 1e-7))

    def test_mna_score_requires_a_solved_level(self):
        mna_advantages = jnp.array([[-0.2, -0.2], [-0.3, -0.3]])
        scores = solved_mna_score(mna_advantages, jnp.array([True, False]))
        np.testing.assert_allclose(np.asarray(scores), np.array([0.5, 0.0]))


if __name__ == "__main__":
    unittest.main()
