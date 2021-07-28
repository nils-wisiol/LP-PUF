import sys

import numpy as np
import scipy as sp
import scipy.stats

import pypuf.attack
import pypuf.batch
import pypuf.io
import pypuf.metrics
import pypuf.simulation
from lppuf import LPPUFv1


def correlation(puf, model):
    return np.array([
        [
            sp.stats.pearsonr(w, v)[0]
            for w in puf.weight_array[:, :-1]
        ]
        for v in model.weight_array[:, :-1]
    ])


class LPPUFv1LRAnalysis(pypuf.batch.StudyBase):

    def parameter_matrix(self):
        return [
            dict(
                n=n,
                noisiness_1=noisiness_1,
                noisiness_2=noisiness_2,
                seed=seed,
                m=m,
                N=N,
                bs=bs,
            )
            for n in [64]
            for noisiness_1 in [0]
            for noisiness_2 in [0]
            for seed in range(2)
            for m, bs, Ns in [
                (1, 1000, [500, 1000, 2000, 5000]),
                (2, 1000, [500, 1000, 5000, 10000, 20000, 50000]),
                (4, 1000, [5000, 10000, 50000, 100000, 200000, 500000]),
                (8, 10**6, [400 * 10**6]),  # RAM: 1M CRPs ~ 4G, 50M ~ 70G, 400M ~ 560G
                (16, 50000, []),
            ]
            for N in Ns
        ]

    def run(self, n, noisiness_1, noisiness_2, m, seed, N, epochs=200, bs=1000):
        # create PUF simulation instance
        puf = LPPUFv1(
            n=n, m=m, noisiness_1=noisiness_1, noisiness_2=noisiness_2,
            seed=pypuf.simulation.base.Simulation.seed(f"LP-PUF LR Experiment Seed {seed}"),
        )

        # generate overall CRPs
        crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N, seed=1)

        # guess CRPs for layer 3
        guessed_layer1_responses = pypuf.io.random_inputs(n=m, N=N, seed=2)
        guessed_layer2_inputs = np.concatenate((crps.challenges, guessed_layer1_responses), axis=1)
        del guessed_layer1_responses
        guessed_layer3_inputs = puf.layer[1].eval(
            guessed_layer2_inputs)  # using puf instance function for convenience, but actually only public info
        del guessed_layer2_inputs
        guessed_layer3_crps = pypuf.io.ChallengeResponseSet(guessed_layer3_inputs, crps.responses)
        del guessed_layer3_inputs

        # run attack on guessed CRPs
        attack = pypuf.attack.LRAttack2021(guessed_layer3_crps, seed=3, k=m, bs=bs, lr=.001, epochs=epochs)
        attack.fit()

        # correct CRPs for layer 3
        correct_layer1_responses = puf.layer[0].eval(crps.challenges)
        correct_layer2_inputs = np.concatenate((crps.challenges, correct_layer1_responses), axis=1)
        correct_layer3_inputs = puf.layer[1].eval(correct_layer2_inputs)

        # compute feature accuracy for layer 3 attack
        guessed_layer3_features = pypuf.simulation.XORArbiterPUF.transform_atf(guessed_layer3_inputs, k=1)[:, 0, :]
        correct_layer3_features = pypuf.simulation.XORArbiterPUF.transform_atf(correct_layer3_inputs, k=1)[:, 0, :]

        # a random unrelated PUF
        puf2 = pypuf.simulation.XORArbiterPUF(
            n=n, k=m,
            seed=pypuf.simulation.base.Simulation.seed(f"LP-PUF LR Experiment Unrelated PUF {seed}"),
        )

        # measure success
        return {
            'correlation_unrelated': correlation(puf2, attack.model),
            'correlation': correlation(puf.layer[2], attack.model),
            'accuracy_unrelated': pypuf.metrics.similarity(puf2, attack.model, seed=31415),
            'accuracy': pypuf.metrics.similarity(puf.layer[2], attack.model, seed=31415),
            'feature_bit_accuracy': np.mean(guessed_layer3_features == correct_layer3_features),
            'feature_vector_accuracy': (guessed_layer3_features == correct_layer3_features).all(axis=1).mean(),
        }


if __name__ == '__main__':
    LPPUFv1LRAnalysis().cli(sys.argv)
