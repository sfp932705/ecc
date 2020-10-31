from pydantic import validate_arguments, StrictInt
from functools import reduce
from typing import List, Tuple


class GF_256():
    """
    Galois Field 256 (2⁸) class that implements the arithmetic needed for Reed-Solomon coding.
    A list of integers is considered to be a polynomial, the first elements being the higher order terms of the polynomial.
    For example, 8x⁶ + 3x⁴ + 240x² + x + 156 is represented as [8,0,3,0,240,1,156].
    The polynomials are represented using lists, although not the most efficient approach, to keep simplicity.
    The coefficients of the polynomial can only take values ranging from 0 to 255.

    The pydantic @validate_arguments decorators are all commented to speed up the Reed-Solomon coding/decoding algorithms.
    """

    # @validate_arguments
    def __init__(self, k: StrictInt = 223):
        """
        Initializes the Galois Field.
        Sets the log and anti-log tables as well as the irreducible and generator polynomials.

        Args:
            k: The max message data length that can be in an encoded Reed-Solomon block using this Finite Field.

        """

        self.n = 255
        self.irreducible = 0x1b  # Irreducible polinomial x⁸+x⁴+x³+x+1 with high order term removed as an integer.
        self.g = 3
        self.gf_256_tables()
        self.generator_pol = self.get_generator(k)

    # @validate_arguments
    def is_zero(self, a: List[StrictInt]) -> bool:
        """
        Checks whether all the coefficients of a polynomial are zero or not.

        Args:
            a: List of integers representing the polynomial.

        Returns:
            True if all coefficients are zero, False otherwise.

        """

        return all([x == 0 for x in a])

    # @validate_arguments
    def get_degree(self, a: List[StrictInt]) -> StrictInt:
        """
        Gets the degree of a polynomial.

        Args:
            a: List of integers representing the polynomial.

        Returns:
            Polynomial degree.

        """

        if a[0] != 0:
            return len(a) - 1
        else:
            for i in range(1, len(a)):
                if a[i] != 0: return len(a) - i - 1
            return 0

    # @validate_arguments
    def get_generator(self, k: StrictInt) -> List[StrictInt]:
        """
        Gets the generator polynomial for the Finite Field.

        Args:
            k: The max message data length that can be in an encoded Reed-Solomon block using this Finite Field.

        Returns:
            Generator polynomial for the Finite Field.

        """

        g = [1]
        for i in range(self.n - k):
            p = [1, self.gf_256_power(self.g, i + 1)]
            g = self.gf_256_pol_multiply(g, p)

        return g

    # @validate_arguments
    def evaluate(self, a: List[StrictInt], x: StrictInt) -> StrictInt:
        """
        Evaluates a polynomial at a given value.
         Args:
             a: List of integers representing the polynomial.

             x: Value at which to evaluate the given polynomial.

         Returns:
            Result of evaluating the polynomial at the given value.

         """

        res = a[-1]
        for _pow, _coef in enumerate(range(len(a) - 2, -1, -1), 1):
            res = self.gf_256_sum(res, self.gf_256_multiply(a[_coef], self.gf_256_power(x, _pow)))

        return res

    # @validate_arguments
    def mul_operation(self, a: StrictInt, b: StrictInt) -> StrictInt:
        """
        Multiplication operation in the Finite Field.

        Args:
            a: Multiplicand.

            b: Multiplier.

        Returns:
            p: Multiplication result.

        """

        p = 0
        for i in range(8):
            if a == 0 or b == 0: break
            if b & 1:
                p ^= a
            b = b >> 1
            carry = a & 0x80
            a = (a << 1) & 0xFF  # To keep lower 8 bits
            if carry:
                a ^= self.irreducible

        return p

    def gf_256_tables(self):
        """
        Builds the log and anti-log tables that will be used in multiplication and division to speed up the operations.
        """

        self.log_table = (self.n + 1) * [0]
        self.antilog_table = self.n * [0]
        x = 1
        for i in range(self.n):
            self.log_table[x] = i
            self.antilog_table[i] = x
            x = self.mul_operation(x, self.g)

    # @validate_arguments
    def gf_256_sum(self, a: StrictInt, b: StrictInt) -> StrictInt:
        """
        Addition operation in the Finite Field.

        Args:
            a: Summand.

            b: Summand.

        Returns:
            Sum result.

        """

        return a ^ b

    # @validate_arguments
    def gf_256_subtract(self, a: StrictInt, b: StrictInt) -> StrictInt:
        """
        Subtraction operation. Same as addition.
        """

        return self.gf_256_sum(a, b)

    # @validate_arguments
    def gf_256_multiply(self, a: StrictInt, b: StrictInt) -> StrictInt:
        """
        Multiply two numbers in Galois Field 256 using the precomputed log and anti-log tables.

        Args:
            a: Multiplicand.

            b: Multiplier.

        Returns:
            Multiplication result.

        """

        if (a == 0) or (b == 0): return 0
        x = self.log_table[a]
        y = self.log_table[b]
        log_mult = (x + y) % self.n

        return self.antilog_table[log_mult]

    # @validate_arguments
    def gf_256_divide(self, a: StrictInt, b: StrictInt):
        """
        Divides two numbers in Galois Field 256 using the precomputed log and anti-log tables.

        Args:
            a: Dividend.

            b: Divisor.

        Returns:
            Division result.

        """

        if (a == 0) or (b == 0): return 0
        x = self.log_table[a]
        y = self.log_table[b]
        log_mult = (x - y) % self.n

        return self.antilog_table[log_mult]

    # @validate_arguments
    def gf_256_power(self, a: StrictInt, n: StrictInt):
        """
        Exponentiation operation in Galois Field 256 using the precomputed log and anti-log tables.

        Args:
            a: Base.

            b: Exponent.

        Returns:
            Exponentiation result.

        """

        x = self.log_table[a]
        z = (x * n) % self.n

        return self.antilog_table[z]

    # @validate_arguments
    def gf_256_pol_sum(self, a: List[StrictInt], b: List[StrictInt]) -> List[StrictInt]:
        """
        Adds two polynomials in Galois Field 256.

        Args:
            a: Summand polynomial.

            b: Summand polynomial.

        Returns:
            Sum result.

        """

        pad = (max(len(a), len(b)) - min(len(a), len(b))) * [0]
        if len(a) > len(b):
            b = pad + b
        elif len(b) > len(a):
            a = pad + a

        return [a[i] ^ b[i] for i in range(len(a))]

    # @validate_arguments
    def gf_256_pol_subtract(self, a: List[StrictInt], b: List[StrictInt]) -> List[StrictInt]:
        """
        Same as polynomial addition.
        """

        return self.gf_256_pol_sum(a, b)

    # @validate_arguments
    def gf_256_pol_multiply(self, a: List[StrictInt], b: List[StrictInt]) -> List[StrictInt]:
        """
        Multiply two polynomials in Galois Field 256.

        Args:
            a: Multiplicand polynomial.

            b: Multiplier polynomial.

        Returns:
            Multiplication result.

        """

        res = [0] * (len(a) + len(b) - 1)
        for o1, i1 in enumerate(a):
            for o2, i2 in enumerate(b):
                mul = self.gf_256_multiply(i1, i2)
                res[o1 + o2] = self.gf_256_sum(res[o1 + o2], mul)

        return res

    # @validate_arguments
    def gf_256_pol_divide(self, a: List[StrictInt], b: List[StrictInt]) -> List[StrictInt]:
        """
        Divides two polynomials in Galois Field 256.

        Args:
            a: Dividend polynomial.

            b: Divisor polynomial.

        Returns:
            Division result.

        """

        return self.gf_256_pol_mod(a, b)[0]

    # @validate_arguments
    def gf_256_pol_power(self, a: List[StrictInt], n: StrictInt) -> List[StrictInt]:
        """
        Polynomial exponentiation in Galois Field 256.

        Args:
            a: Base polynomial.

            b: Exponent.

        Returns:
            Polynomial exponentiation result.

        """

        b = a[:]
        for i in range(n - 1):
            a = self.gf_256_pol_multiply(a, b)

        return a

    # @validate_arguments
    def gf_256_pol_mod(self, a: List[StrictInt], b: List[StrictInt]) -> Tuple[List[StrictInt], List[StrictInt]]:
        """
        Polynomial modular division in Galois Field 256.

        Args:
            a: Dividend polynomial.

            b: Divisor polynomial.

        Returns:
            Quotient and remainder polynomials.

        """

        a_degree, b_degree = self.get_degree(a), self.get_degree(b)
        a_coeff, b_coeff = a[-(a_degree + 1)], b[-(b_degree + 1)]
        if b_degree < 0:
            raise ZeroDivisionError
        elif a_degree < b_degree:
            quotient, remainder = [0], a
        else:
            quotient, remainder = (a_degree + 1) * [0], a
            quotient_degree, remainder_degree, = a_degree - b_degree, a_degree
            remainder_coeff = a_coeff
            while quotient_degree >= 0 and not self.is_zero(remainder):
                quotient_coeff = self.gf_256_divide(remainder_coeff, b_coeff)
                q = [quotient_coeff] + [0] * quotient_degree
                quotient[-(quotient_degree + 1)] = quotient_coeff
                remainder = self.gf_256_pol_subtract(remainder, self.gf_256_pol_multiply(q, b))
                remainder_degree = self.get_degree(remainder)
                remainder_coeff = remainder[-(remainder_degree + 1)]
                quotient_degree = remainder_degree - b_degree

        return quotient, remainder


class RS():
    """Implements Reed-Solomon error correction with blocks of length 255."""

    @validate_arguments
    def __init__(self, k: StrictInt = 223):
        """
        Sets the Galois Field instance.

        Args:
            k: Max  message length. In each block, the last 255 - k  will be parity coefficients.

        """

        self.ff = GF_256(k)
        self.n = self.ff.n
        self.k = k

    @validate_arguments
    def encode(self, pol: List[StrictInt]) -> List[StrictInt]:
        """
        Encodes a polynomial. The polynomial is decomposed into blocks of polynomials of max length of self.k - 1.
        In each block, the last coefficient holds the length of the block.

        Args:
            pol: Polynomial representing the data to be encoded.

        Returns:
            Encoded polynomial representing the data.

        """

        enc = []
        parity = (self.n - self.k) * [0]
        for i in range(0, len(pol), self.k - 1):
            enc += self.block_encode(pol[i:i + self.k - 1] + [len(pol[i:i + self.k - 1])] + parity)

        return enc

    @validate_arguments
    def block_encode(self, block: List[StrictInt]) -> List[StrictInt]:
        """
        Encodes a block of data.

        Args:
            block: Polynomial representing the block of data to be encoded.

        Returns:
            Polynomial representing the block of encoded data.

        """

        b = self.ff.gf_256_pol_mod(block, self.ff.generator_pol)[1]

        return self.ff.gf_256_pol_subtract(block, b)

    @validate_arguments
    def decode(self, pol: List[StrictInt]) -> List[StrictInt]:
        """
        Decodes a polynomial composed of blocks which are each at most 255 coefficients long.

        Args:
            pol: Polynomial representing the data to be decoded.

        Returns:
            Decoded polynomial representing the data.

        """

        dec = []
        for i in range(0, len(pol), self.n):
            dec += self.block_decode(pol[i:i + self.n])

        return dec

    @validate_arguments
    def block_decode(self, block: List[StrictInt]) -> List[StrictInt]:
        """
        Decodes a block of data.

        Args:
            block: Polynomial representing the block of data to be decoded.

        Returns:
            Polynomial representing the block of decoded data.

        """

        syndromes = self.syndromes(block)
        corrected = self.correct(block, syndromes)[0]
        _len = corrected.pop()

        return corrected[-_len:]

    @validate_arguments
    def data_parity_split(self, pol: List[StrictInt]) -> Tuple[List[StrictInt], List[StrictInt]]:
        """
        Splits a polynomial into data coefficients and parity coefficients. The last 255 - k coefficients are parity coefficients.

        Args:
            pol: Polynomial to be split into data and parity polynomials.

        Returns:
            Data and parity polynomials.

        """

        partition = -(self.n - self.k)
        data, parity = pol[:partition], pol[partition:]

        return data, parity

    @validate_arguments
    def syndromes(self, pol: List[StrictInt]) -> List[StrictInt]:
        """
        Calculates polynomial syndromes.

        Args:
            pol: Polynomial whose syndromes are to be calculated.

        Returns:
            Syndromes for the given polynomial.

        """

        L = range(self.n - self.k - 1, -1, -1)
        return [self.ff.evaluate(pol, self.ff.gf_256_power(self.ff.g, l + 1)) for l in L] + [0]

    @validate_arguments
    def correct(self, pol: List[StrictInt], syndromes: List[StrictInt]) -> Tuple[List[StrictInt], List[StrictInt]]:
        """
        Corrects a polynomial given its syndromes.

        Args:
            pol: Polynomial to be corrected.

            syndromes: Syndromes of the polynomial.

        Returns:
            Corrected polynomial split in data and parity polynomials.

        Raises:
            Exception: if too many errors occurred and correct decoding is impossible.

        """

        if self.ff.is_zero(syndromes):
            return self.data_parity_split(pol)
        else:
            sigma, omega = self.sigma_omega(syndromes)
            X, j = self.error_evaluator(sigma)
            Y = self.forney(omega, X)

        Elist = [0] * self.n
        if len(Y) >= len(j):
            for i in range(self.n):
                if i in j:
                    Elist[i] = Y[j.index(i)]
            E = Elist[::-1]
        else:
            E = []

        c = self.ff.gf_256_pol_subtract(pol, E)
        if self.ff.get_degree(c) > self.ff.get_degree(pol):
            raise Exception("Too many errors.")

        return self.data_parity_split(c)

    @validate_arguments
    def sigma_omega(self, syndromes: List[StrictInt]) -> Tuple[List[StrictInt], List[StrictInt]]:
        """
        Computes the error locator polynomial and the error evaluator polynomial.

        Args:
            syndormes: Polynomial syndromes.

        Returns:
            sigma: Error locator polynomial.

            omega: Error evaluator polynomial.

        """

        sigma, old_sigma, omega, old_omega, A, old_A, B, old_B = [1], [1], [1], [1], [0], [0], [1], [1]
        L, old_L, M, old_M = 0, 0, 0, 0
        synd_shift = 0
        pol_multiply, pol_subtract = self.ff.gf_256_pol_multiply, self.ff.gf_256_pol_subtract
        pol_mod, n, k = self.ff.gf_256_pol_mod, self.n, self.k
        if len(syndromes) > (n - k): synd_shift = len(syndromes) - (n - k)
        erasures_count = 0
        for l in range(0, n - k - erasures_count):
            K = erasures_count + l + synd_shift
            Delta = [self.ff.gf_256_pol_multiply(syndromes[:-1] + [1], old_sigma)[-K - 1]]
            sigma = pol_subtract(old_sigma, pol_multiply(pol_multiply(Delta, [1, 0]), old_B))
            omega = pol_subtract(old_omega, pol_multiply(pol_multiply(Delta, [1, 0]), old_A))
            if Delta == [0] or 2 * old_L > K + erasures_count or (2 * old_L == K + erasures_count and old_M == 0):
                A, B, L, M = pol_multiply([1, 0], old_A), pol_multiply([1, 0], old_B), old_L, old_M
            elif (Delta != [0] and 2 * old_L < K + erasures_count) or (2 * old_L == K + erasures_count and old_M != 0):
                A, B, L, M = pol_mod(old_omega, Delta)[0], pol_mod(old_sigma, Delta)[0], K - old_L, 1 - old_M

            old_sigma, old_omega, old_A, old_B = sigma, omega, A, B

        if self.ff.get_degree(omega) > self.ff.get_degree(sigma):
            omega = [omega[-(self.ff.get_degree(sigma) + 1):]]

        omega = pol_mod(pol_multiply(syndromes, sigma), [1] + (n - k + 1) * [0])[1]

        return sigma, omega

    @validate_arguments
    def error_evaluator(self, sigma: List[StrictInt]) -> Tuple[List[StrictInt], List[StrictInt]]:
        X, j = [], []
        for l in range(1, self.n + 1):
            if self.ff.evaluate(sigma, self.ff.gf_256_power(self.ff.g, l)) == 0:
                X.append(self.ff.gf_256_power(self.ff.g, -l))
                j.append(self.n - l)
        if len(j) != self.ff.get_degree(sigma):
            raise Exception("Error evaluator check not passed.")

        return X, j

    @validate_arguments
    def forney(self, omega: List[StrictInt], X: List[StrictInt]) -> List[StrictInt]:
        """
        Forney algorithm from https://en.wikipedia.org/wiki/Forney_algorithm with c = 1, so the expression is simplified.
        """

        E, X_range = [], range(len(X))
        for i in X_range:
            inv = self.ff.gf_256_divide(1, X[i])
            s_tmp = [self.ff.gf_256_subtract(1, self.ff.gf_256_multiply(inv, X[j])) for j in X_range if j != i]
            sigma_prime = reduce(self.ff.gf_256_multiply, s_tmp, 1)
            E.append(self.ff.gf_256_subtract(0, self.ff.gf_256_divide(self.ff.evaluate(omega, inv), sigma_prime)))

        return E
