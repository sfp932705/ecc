import bitarray
from pydantic import validate_arguments, StrictInt
from typing import List


class bit_array():
    """Bit array validator class."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not isinstance(v, bitarray.bitarray):
            raise TypeError('bitarray.bitarray required')
        return v


class Message():
    """
    Message class that will be used to test error correcting codes.
    """

    @validate_arguments
    def __init__(self, encoding: str = 'utf-8') -> None:
        """
        Stores Message encoding attribute and creates the mapping used for uint8 to bit and vice-versa conversion.

        Args:
            encoding: Encoding to be used.
        """

        self.encoding = encoding
        self.int2bit_mapping = {x: bitarray.bitarray('{:08b}'.format(x)) for x in range(256)}

    @validate_arguments
    def to_bits(self, msg: str) -> bit_array:
        """
        Encodes a string as a bit array using the encoding attribute.

        Args:
            msg: String to be encoded.

        Returns:
            ba: Bit array containing encoded string.
        """

        ba = bitarray.bitarray()
        ba.frombytes(msg.encode(self.encoding))

        return ba

    @validate_arguments
    def from_bits(self, bits: bit_array) -> str:
        """
        Decodes a bit array into a string using the encoding attribute.

        Args:
            bits: Bit array to be decoded.

        Returns:
            Decoded string.
        """

        return bits.tobytes().decode(self.encoding)

    @validate_arguments
    def bits2int(self, bits: bit_array) -> List[StrictInt]:
        """
        Converts a bit array into a list of integers. Each integer is decoded after
        8 consecutive bits, therefore all range from 0 to 255.

        Args:
            bits: Bit array to be converted to list of integers.

        Returns:
            List of integers.
        """

        assert len(bits) % 8 == 0, ' Length of sequence of bits must be a multiple of 8.'

        return bits.decode(self.int2bit_mapping)

    @validate_arguments
    def int2bits(self, uints8: List[StrictInt]) -> bit_array:
        """
        Converts a list of integers (each taking values from 0 to 255) to a bitarray.
        Each integer is encoded in 8 bits.

        Args:
            uints8: List of integers that will be mapped to a bitarray.

        Returns:
            Bitarray.
        """

        ba = bitarray.bitarray()
        ba.encode(self.int2bit_mapping, uints8)

        return ba

    @validate_arguments
    def negate(self, bits: bit_array, *args: StrictInt) -> bit_array:
        """
        Negates the specified indexes of a given bit array.

        Args:
            bits: Bit array to be altered.

            *args: Indexes of the bits to be negated.

        Returns:
            altered: Bit array with specified indexes negated.
        """

        altered = bits[:]
        try:
            [altered.invert(x) for x in args]
        except IndexError:
            error = f'Error when altering message. Valid positions range from 0 to {len(bits) - 1}.'
            raise ValueError(error) from IndexError

        return altered
