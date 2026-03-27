import array
import struct

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend ke depan
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # bit awal pada byte terakhir diganti 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)


class EliasGammaPostings:
    """
    Kompresi Elias Gamma untuk integer positif.

    Untuk postings list, data disimpan dalam bentuk gap seperti VBEPostings.
    Untuk tf list, data disimpan apa adanya (tanpa gap).
    """

    @staticmethod
    def _to_gamma_bits(number):
        if number <= 0:
            raise ValueError("Elias gamma hanya menerima integer positif")
        binary = bin(number)[2:]
        return ("0" * (len(binary) - 1)) + binary

    @staticmethod
    def _pack_bits_to_bytes(bitstring):
        if not bitstring:
            return b""
        padded_len = ((len(bitstring) + 7) // 8) * 8
        bitstring = bitstring.ljust(padded_len, "0")
        payload = bytearray()
        for i in range(0, padded_len, 8):
            payload.append(int(bitstring[i:i+8], 2))
        return bytes(payload)

    @staticmethod
    def _unpack_bytes_to_bits(bytestream):
        return "".join(format(b, "08b") for b in bytestream)

    @staticmethod
    def _encode_number_list(list_of_numbers):
        if not list_of_numbers:
            return struct.pack(">I", 0)
        bitstring = "".join(EliasGammaPostings._to_gamma_bits(n) for n in list_of_numbers)
        return struct.pack(">I", len(list_of_numbers)) + EliasGammaPostings._pack_bits_to_bytes(bitstring)

    @staticmethod
    def _decode_number_list(encoded_stream):
        if not encoded_stream:
            return []

        count = struct.unpack(">I", encoded_stream[:4])[0]
        if count == 0:
            return []

        bits = EliasGammaPostings._unpack_bytes_to_bits(encoded_stream[4:])
        numbers = []
        i = 0
        while len(numbers) < count and i < len(bits):
            zero_count = 0
            while i < len(bits) and bits[i] == "0":
                zero_count += 1
                i += 1

            if i >= len(bits):
                break

            total_len = zero_count + 1
            if i + total_len > len(bits):
                break

            value_bits = bits[i:i + total_len]
            numbers.append(int(value_bits, 2))
            i += total_len

        return numbers

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes dengan Elias Gamma.
        Postings diubah dulu menjadi gap-based list.
        """
        if len(postings_list) == 0:
            return struct.pack(">I", 0)
        gap_postings_list = [postings_list[0] + 1]
        for i in range(1, len(postings_list)):
            gap_postings_list.append((postings_list[i] - postings_list[i - 1]) + 1)
        return EliasGammaPostings._encode_number_list(gap_postings_list)

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode postings_list dari encoded Elias Gamma lalu rekonstruksi dari gap.
        """
        gap_postings_list = EliasGammaPostings._decode_number_list(encoded_postings_list)
        if len(gap_postings_list) == 0:
            return []
        postings_list = [gap_postings_list[0] - 1]
        for i in range(1, len(gap_postings_list)):
            postings_list.append(postings_list[-1] + (gap_postings_list[i] - 1))
        return postings_list

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list TF apa adanya (tanpa gap) dengan Elias Gamma.
        """
        return EliasGammaPostings._encode_number_list(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode list TF dari encoded Elias Gamma.
        """
        return EliasGammaPostings._decode_number_list(encoded_tf_list)

if __name__ == '__main__':
    
    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("byte hasil encode postings: ", encoded_postings_list)
        print("ukuran encoded postings   : ", len(encoded_postings_list), "bytes")
        print("byte hasil encode TF list : ", encoded_tf_list)
        print("ukuran encoded postings   : ", len(encoded_tf_list), "bytes")
        
        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("hasil decoding (postings): ", decoded_posting_list)
        print("hasil decoding (TF list) : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, "hasil decoding tidak sama dengan postings original"
        assert decoded_tf_list == tf_list, "hasil decoding tidak sama dengan postings original"
        print()
