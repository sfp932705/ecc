from functools import reduce
import bitarray
from pydantic import validate_arguments, StrictInt
from .message import bit_array


class Hamming():
    """Hamming error corrector."""

    @validate_arguments
    def __init__(self, m: StrictInt) -> None:
        """
        Builds a hamming dict that contains the number of parity bits desired to encode a block
        and the maximum data bits that can be encoded with this amount of parity bits.

        Args:
            m: Parity bits desired.
        """

        self.hamming_dict = {'m': m, 'k': 2 ** m - m - 1}

    @validate_arguments
    def interlace(self, bits: bit_array, inverse: bool = False) -> bit_array:
        """
        Interlaces a sequence of bits composed of n blocks. Takes bits, one at a time,
        from each block, and appends them to a new sequence of bits.

        Args:
            bits: Sequence of bits to be interlaced.

            inverse: Whether to apply inverse interlace transformation or not.

        Returns:
            interlaced: Sequence of interlaced bits.
        """

        interlaced = bitarray.bitarray()
        blocks = len(bits) // 2 ** self.hamming_dict['m']
        if not inverse:
            for i in range(2 ** self.hamming_dict['m']):
                interlaced += [bits[i + j * 2 ** self.hamming_dict['m']] for j in range(blocks)]
        else:
            for i in range(blocks):
                interlaced += [bits[i + j * blocks] for j in range(2 ** self.hamming_dict['m'])]

        return interlaced

    @validate_arguments
    def block_encode(self, bits: bit_array) -> bit_array:
        """
        Encodes a block of data bits.

        Args:
            bits: Block of data bits to be encoded.

        Returns:
            bits: Encoded block of bits.
        """

        bits = bits[:]
        # Insert 0 at position 0.
        bits.insert(0, 0)
        # Insert 1 into parity bits positions.
        [bits.insert(2 ** x, 1) for x in range(self.hamming_dict['m'])]
        # Encode data.
        xor = reduce(lambda x, y: x ^ y, [i for i, bit in enumerate(bits) if int(bit) == 1])
        negate = [2 ** i for i, bit in enumerate(bin(xor)[:1:-1]) if int(bit) == 1]
        [bits.invert(x) for x in negate]
        # Guarantee even parity check.
        if bits.count(1) % 2 == 1:
            bits[0] = 1

        return bits

    @validate_arguments
    def encode(self, bits: bit_array) -> bit_array:
        """
        Encodes a sequence of data bits. The data is divided in as many blocks
        as needed. The last block of data is filled if necessary to complete the required block size.
        Afterwards, an information block containing the number of bits filled in the last block is prepended.
        Finally, the blocks are interlaced since in practice errors tend to come in bursts.

        Args:
            bits: Sequence of data bits to be encoded.

        Returns:
            interlaced: Sequence of encoded and interlaced bits.
        """

        m = len(bits)
        enc_bits = bitarray.bitarray()
        filled = 0
        for i in range(0, m, self.hamming_dict['k']):
            _bits = bits[i:i + self.hamming_dict['k']]
            if len(_bits) != self.hamming_dict['k']:
                # Fill rest of block with 0.
                filled = self.hamming_dict['k'] - len(_bits)
                _bits += bitarray.bitarray(filled * [0])

            # Encode bits
            enc_bits += self.block_encode(_bits)
        # Add information block.
        info = bitarray.bitarray(bin(filled)[2:])
        info = bitarray.bitarray((self.hamming_dict['k'] - len(info)) * [0]) + info
        enc_bits = self.block_encode(info) + enc_bits
        # Interlace bits from each block.
        interlaced = self.interlace(enc_bits)

        return interlaced

    @validate_arguments
    def block_decode(self, bits: bit_array) -> bit_array:
        """
        Decodes a block of bits.

        Args:
            bits: Block of bits to be decoded.

        Returns:
            bits: Decoded block of data bits.
        """

        bits = bits[:]
        try:
            xor = reduce(lambda x, y: x ^ y, [i for i, bit in enumerate(bits) if int(bit) == 1])
            if xor != 0:
                bits.invert(xor)
            if bits.count(1) % 2 == 1 and xor != 0:
                print('\nERROR!!! \nMore than one error detected in same block. Unable to correct.\n')
        except TypeError:
            # This happens when all bits in received block are zero.
            pass
        [bits.pop(2 ** x) for x in range(self.hamming_dict['m'] - 1, -1, -1)]
        bits.pop(0)

        return bits

    @validate_arguments
    def decode(self, bits: bit_array) -> bit_array:
        """
        Decodes a sequence of bits. The bits are passed through an inverse
        interlaced transformation. Each block is decoded.

        Args:
            bits: Sequence of bits to be decoded.

        Returns:
            dec_bits: Sequence of decoded bits.
        """

        # Inverse interlace bits from different blocks.
        bits = self.interlace(bits, True)
        dec_bits = bitarray.bitarray()
        m = len(bits)
        for i in range(0, m, 2 ** self.hamming_dict['m']):
            _bits = bits[i:i + 2 ** self.hamming_dict['m']]
            if i == 0:
                # Information block.
                filled = int(self.block_decode(_bits).to01(), 2)
            elif i == (m - 2 ** self.hamming_dict['m']):
                # Last data block.
                dec_bits += self.block_decode(_bits)[:self.hamming_dict['k'] - filled]
            else:
                # Data blocks.
                dec_bits += self.block_decode(_bits)

        return dec_bits
