import numpy as np
import pypuf.simulation
import pypuf.simulation.base
import pypuf.io


class SplitChallengeBlock(pypuf.simulation.base.Simulation):
    """
    In this block, the ``n``-bit challenge is split in ``m`` consecutive partitions. Each partitioned challenge
    is applied to an individual Arbiter PUF (with challenge length ``n//m``), the resulting ``m`` challenge bits are
    returned without further processing.

    The Arbiter PUFs are chosen independently of each other.
    """

    def __init__(self, n, seed, m, noisiness=0):
        assert n % m == 0, f'n must be multiple of m={m}, got {n}'
        assert n > 0, f'n must be positive, got {n}'
        self.n = n
        self.o = n // m
        self.pufs = [
            pypuf.simulation.ArbiterPUF(
                n=self.o,
                seed=self.seed(f'{self.__class__} seed={seed} {i}'),
                transform=pypuf.simulation.ArbiterPUF.transform_id,
                noisiness=noisiness,
            )
            for i in range(m)
        ]

    @property
    def challenge_length(self) -> int:
        return len(self.pufs) * self.pufs[0].challenge_length

    @property
    def response_length(self) -> int:
        return len(self.pufs)

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        assert challenges.shape[1] == self.n, f'Challenges must have length {self.n}, got {challenges.shape[1]}.'
        N, n = challenges.shape
        responses = np.empty(shape=(N, n // self.o))
        for i in range(0, self.n, self.o):
            responses[:, i // self.o] = self.pufs[i // self.o].eval(challenges[:, i:i + self.o])
        return responses


class TBlock(pypuf.simulation.base.Simulation):
    """
    This block takes ``n1 + n2`` challenge bits and returns ``n1`` response bits.

    It's purpose is to "spread" the ``n2`` bits across the ``n1`` response bits to make the prediction of these response
    bits hard for anyone with limited knowledge of the ``n2`` input bits. The ``n1`` challenge bits, in contrast, are
    not "spread".
    """

    def __init__(self, n1, n2, seed):
        """
        n1 "public" bits
        n2 "secret" bits
        """
        super().__init__()
        self.n1, self.n2 = n1, n2
        rng = np.random.default_rng(seed)
        self.parities_of = [
            rng.choice(a=n2, size=max(n2 // 2, 1), replace=False)
            for _ in range(n1)
        ]

    @property
    def challenge_length(self) -> int:
        return self.n1 + self.n2

    @property
    def response_length(self) -> int:
        return self.n1

    @staticmethod
    def public_parity(pos, challenges):
        return challenges[:, pos]

    def private_parity(self, pos, challenges):
        parity_of = self.parities_of[pos]
        inputs = challenges[:, self.n1:]
        return np.cumprod(inputs[:, parity_of], axis=1)[:, -1]

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        responses = np.empty(shape=(challenges.shape[0], self.n1))

        for i in range(self.n1):
            responses[:, i] = (
                    self.public_parity(pos=i, challenges=challenges) *
                    self.private_parity(pos=i, challenges=challenges)
            )

        return responses


class LPPUFv1(pypuf.simulation.base.Simulation):
    """
    A Arbiter PUF-based PUF design, composed of three layers:
    1. SplitChallengeBlock with ``n`` input bits and ``m`` partitions, taking the input as input
    2. TBlock taking the output of layer 1 as private and the input as public input
    3. XORArbiterPUF with ``m`` Arbiter PUFs, taking the output of layer 2 as input.

    All in all, it takes ``n`` challenge bits and returns a single response bit.
    """

    def __init__(self, n, seed, m, noisiness_1=0, noisiness_2=0):
        super().__init__()
        self.layer = [
            SplitChallengeBlock(n=n, m=m, noisiness=noisiness_1, seed=self.seed(f'LPPUFv1 {seed} Layer 1')),
            TBlock(n1=n, n2=m, seed=self.seed(f'LPPUFv1 {seed} Layer 2')),
            pypuf.simulation.XORArbiterPUF(n=n, k=m, noisiness=noisiness_2, seed=self.seed(f'LPPUFv1 {seed} Layer 3')),
        ]

    @property
    def challenge_length(self) -> int:
        return self.layer[0].challenge_length

    @property
    def response_length(self) -> int:
        return self.layer[-1].response_length

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        responses = self.layer[0].eval(challenges)
        responses = self.layer[1].eval(np.concatenate((challenges, responses), axis=1))
        responses = self.layer[2].eval(responses)
        return responses
