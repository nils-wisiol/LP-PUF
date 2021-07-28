import sys

import pypuf.attack
import pypuf.batch
import pypuf.metrics

from lppuf import LPPUFv1


class LPPUFv1MLPFullAnalysis(pypuf.batch.StudyBase):

    def parameter_matrix(self):
        return [
            dict(
                n=n,
                noisiness_1=noisiness_1,
                noisiness_2=noisiness_2,
                num=num,
                seed=seed,
                m=m,
                N=N,
                epochs=50,
                bs=1000,
                lr=.002,
                early_stop=.08,
                net=net,
            )
            for n in [64]
            for noisiness_1 in [0]  # .01,.02,.05,.1, .2, .35, .5]
            for noisiness_2 in [0]  # .01,.02,.05,.1, .2, .35, .5]
            for num in [10]
            for seed in range(10)
            for m, net, Ns in [
                # (1, [2**4, 2**5, 2**4], [100000, 200000, 500000]),
                # (2, [2**4, 2**5, 2**4], [100000, 200000, 500000]),
                (4, [2**4, 2**5, 2**4], [
                    # 200000, 300000, 400000, 500000, 10**6,
                    # 10 * 10**6,  # 11G
                    50 * 10**6,
                ]),
                (4, [2**5, 2**6, 2**5], [
                    # 200000, 300000, 400000, 500000, 10**6,
                    10 * 10**6,  # 11G
                    50 * 10**6,
                ]),
                # (8, [2**8, 2**9, 2**8], [150 * 10**6, 350 * 10**6, 500 * 10**6]),
                (16, [], []),
            ]
            for N in Ns
        ]

    def run(self, n, noisiness_1, noisiness_2, num, m, seed, N, epochs, bs, lr, early_stop, net):
        # create PUF simulation instance
        puf = LPPUFv1(
            n=n, m=m, noisiness_1=noisiness_1, noisiness_2=noisiness_2,
            seed=pypuf.simulation.base.Simulation.seed(f"LP-PUF MLP Experiment Seed {seed}"),
        )

        # generate overall CRPs
        crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N, seed=1)

        # run attack on guessed CRPs
        attack = pypuf.attack.MLPAttack2021(
            crps, seed=2, net=net,
            bs=bs, lr=lr, epochs=epochs, early_stop=early_stop,
        )
        attack.fit()

        # measure success
        return {
            'accuracy': pypuf.metrics.similarity(puf, attack.model, seed=31415),
            'history': attack.history,
            'model': attack.model.weights,
        }


if __name__ == '__main__':
    LPPUFv1MLPFullAnalysis().cli(sys.argv)
