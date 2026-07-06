import random
import unittest

import numpy as np

import cslearn.cstree as ct
import cslearn.stage as st


class TestingToMinimalCSIs(unittest.TestCase):
    def test_v_structure(self):
        # 3 variables, 2 outcomes each
        tree = ct.CStree([2] * 3, labels=["a", "b", "c"])

        # V-structure
        tree.update_stages(
            {
                0: [st.Stage([{0, 1}])],
                1: [
                    st.Stage([0, 0]),
                    st.Stage([0, 1]),
                    st.Stage([1, 0]),
                    st.Stage([1, 1]),
                ],
            }
        )
        minl_csis = tree.to_minimal_context_csis()

        csi_strings = set([])
        for cont, csis in minl_csis.items():
            for csi in csis:
                csi_strings.add(str(csi))

        # Testing against the string representation of the CSIs.
        # Maybe not the best, but quite good anyway since it
        # is easy to read and understand.
        correct_csis = {"a ⊥ b"}
        self.assertEqual(csi_strings, correct_csis)

    def test_figure1(self):
        tree = ct.CStree([2, 2, 2, 2], labels=["X" + str(i) for i in range(1, 5)])
        tree.update_stages(
            {
                0: [st.Stage([0]), st.Stage([1])],
                1: [
                    st.Stage([{0, 1}, 0], color="green"),
                    st.Stage([0, 1]),
                    st.Stage([1, 1]),
                ],
                2: [
                    st.Stage([0, {0, 1}, 0], color="blue"),
                    st.Stage([0, {0, 1}, 1], color="orange"),
                    st.Stage([1, {0, 1}, 0], color="red"),
                    st.Stage([1, 1, 1]),
                    st.Stage([1, 0, 1]),
                ],
            }
        )

        csi_strings = set([])
        minl_csis = tree.to_minimal_context_csis()
        for cont, csis in minl_csis.items():
            for csi in csis:
                csi_strings.add(str(csi))

        correct_csis = {"X1 ⊥ X3 | X2=0", "X2 ⊥ X4 | X1, X3=0", "X2 ⊥ X4 | X3, X1=0"}
        self.assertEqual(csi_strings, correct_csis)

    def test_singleton_tree_has_no_nontrivial_csis(self):
        """A fully singleton tree (no context-specific stages) has no non-trivial CSI relations."""
        np.random.seed(0)
        random.seed(0)
        tree = ct.sample_cstree([2, 2, 2], max_cvars=0, prob_cvar=0.0, prop_nonsingleton=0)
        rels = tree.csi_relations_per_level()
        for level, level_rels in rels.items():
            nontrivial = [r for r in level_rels if len(r.context.context) > 0]
            self.assertEqual(nontrivial, [], f"unexpected CSI at level {level}")


if __name__ == "__main__":
    unittest.main()
