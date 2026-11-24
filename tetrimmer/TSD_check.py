import math
from collections import Counter

import numpy as np
from Bio import AlignIO


def log10_binomial_tail(k, n, p):
    """
    Calculate the probability to get at least k successes in n trials.
    :param k: int, number of successes
    :param n: int, number of trials
    :param p: float (0-1), success probability of one trial
    :return: float, log10 of P(X >= k)
    """
    if k <= 0 or p >= 1:
        return 0.0
    if p <= 0:
        return -math.inf
    terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        + i * math.log(p) + (n - i) * math.log1p(-p)
        for i in range(k, n + 1)
    ]
    largest = max(terms)
    log_tail = largest + math.log(sum(math.exp(term - largest) for term in terms))
    return min(log_tail / math.log(10), 0.0)


def is_low_complexity(kmer):
    """
    Check if a k-mer is a homopolymer or a dinucleotide repeat like ATAT.
    """
    if len(set(kmer)) == 1:
        return True
    if len(kmer) >= 4 and kmer == (kmer[:2] * len(kmer))[: len(kmer)]:
        return True
    return False


NUCLEOTIDES = [b'A', b'C', b'G', b'T']


def read_msa(msa_file):
    """
    Read an MSA as a matrix. Each row represents a sequence and each column represents a position.
    :param msa_file: str, path to the MSA
    :return: list of sequence ids and numpy array of upper case characters
    """
    alignment = AlignIO.read(msa_file, 'fasta')
    msa = np.array(
        [
            np.frombuffer(str(record.seq).upper().encode(), dtype='S1')
            for record in alignment
        ]
    )
    return [record.id for record in alignment], msa


def nucleotide_counts(msa):
    """
    Count A, C, G and T in every column.
    :return: numpy array with one row for each nucleotide
    """
    return np.array([(msa == nucleotide).sum(axis=0) for nucleotide in NUCLEOTIDES])


def consensus_columns(msa, gap_threshold=0.8, simi_check_gap_thre=0.4, similarity_threshold=0.7, min_nucleotide=5):
    """
    Find the columns that remain after gap removal, the same rules as remove_gaps_with_similarity_check.
    :return: boolean numpy array, True for columns that contribute a position to the consensus sequence
    """
    counts = nucleotide_counts(msa)
    total = counts.sum(axis=0)
    gap_count = (msa == b'-').sum(axis=0)
    gap_fraction = gap_count / msa.shape[0]
    similarity = np.zeros(msa.shape[1])
    np.divide(counts.max(axis=0), total, out=similarity, where=total > 0)
    return (
        (msa.shape[0] - gap_count >= min_nucleotide)
        & (gap_fraction <= gap_threshold)
        & ((gap_fraction < simi_check_gap_thre) | (similarity >= similarity_threshold))
    )


def anchor_end(agreement, end, window=20, search_range=100, min_contrast=0.25, min_agreement=0.5):
    """
    Find the right edge of the conserved block next to a boundary.
    The edge is the position with the largest difference between the mean agreement of the columns before it and
    the mean agreement of the columns after it.

    :param agreement: numpy array, proportion of sequences that share the most common nucleotide of each column.
    Columns in which only few sequences have a nucleotide are not part of this array
    :param end: int, number of columns in agreement that belong to the TE
    :param window: int, number of columns used on each side of a position. Default: 20
    :param search_range: int, the edge is searched this number of columns before and after the boundary. Default: 100
    :param min_contrast: float, minimum difference of the mean agreement at the edge. Default: 0.25
    :param min_agreement: float, minimum agreement of the last column of the TE. Default: 0.5
    :return: int, anchored boundary. None if no edge is found
    """
    cumulative = np.concatenate(([0], np.cumsum(agreement)))
    positions = np.arange(
        max(end - search_range, window), min(end + search_range, len(agreement) - window) + 1
    )
    if len(positions) == 0:
        return None
    inside = (cumulative[positions] - cumulative[positions - window]) / window
    outside = (cumulative[positions + window] - cumulative[positions]) / window
    contrast = inside - outside
    if contrast.max() < min_contrast:
        return None
    edge = int(positions[contrast.argmax()])

    # The TE has to end with a conserved column
    for _ in range(window):
        if agreement[edge - 1] >= min_agreement:
            return edge
        edge -= 1
    return None


def anchor_boundaries(msa, start, end):
    """
    Move both TE boundaries to the edges of the conserved block of the MSA.
    A boundary that lies in the flanking region splits the copies at non-homologous positions, which hides TSDs.

    :param msa: numpy array from read_msa
    :param start: int, first MSA column of the TE (0-based)
    :param end: int, MSA column after the last column of the TE
    :return: tuple (anchored start, anchored end, start edge found, end edge found). A boundary is not changed if no
    edge is found next to it
    """
    margin = min(150, (end - start) // 4)

    def anchor(region_start, region_end, boundary, left):
        # Use the sequences that contain this TE end. They are selected in a region inside the TE
        region = msa[:, region_start:region_end]
        counts = nucleotide_counts(region)
        solid_columns = np.flatnonzero(
            (counts.sum(axis=0) >= 5) & ((region == b'-').sum(axis=0) / msa.shape[0] <= 0.8)
        )
        if len(solid_columns) < 20:
            return None
        majority = np.array(NUCLEOTIDES)[counts.argmax(axis=0)]
        identity = (region[:, solid_columns] == majority[solid_columns]).mean(axis=1)
        present = msa[identity >= 0.5]
        if len(present) < 5:
            return None

        # Ignore columns in which only few of these sequences have a nucleotide, like insertions of single copies
        occupied_columns = np.flatnonzero((present != b'-').sum(axis=0) >= max(2, 0.2 * len(present)))
        agreement = nucleotide_counts(present[:, occupied_columns]).max(axis=0) / len(present)
        te_columns = np.searchsorted(occupied_columns, boundary)

        if left:
            # The left boundary is anchored on the mirrored MSA
            edge = anchor_end(agreement[::-1], len(occupied_columns) - te_columns)
            if edge is None:
                return None
            return int(occupied_columns[len(occupied_columns) - edge])

        edge = anchor_end(agreement, te_columns)
        if edge is None:
            return None
        return int(occupied_columns[edge - 1]) + 1

    anchored_start = anchor(start + margin, min(start + margin + 200, end), start, left=True)
    anchored_end = anchor(max(end - margin - 200, start), end - margin, end, left=False)
    start_found = anchored_start is not None
    end_found = anchored_end is not None
    if not start_found:
        anchored_start = start
    if not end_found:
        anchored_end = end
    if anchored_start >= anchored_end:
        return start, end, False, False
    return anchored_start, anchored_end, start_found, end_found


def collect_flanks(
    ids, msa, start, end, flank_len=30, terminal_win=30, min_terminal_identity=0.6
):
    """
    Extract the genomic sequences directly left and right of the TE for each sequence of the MSA.
    Only sequences that contain both TE ends are used, because the flanks of a truncated copy are unknown.

    :param ids: list of sequence ids from read_msa
    :param msa: numpy array from read_msa, the MSA contains the TE and its flanking regions
    :param start: int, first MSA column of the TE (0-based)
    :param end: int, MSA column after the last column of the TE
    :param flank_len: int, number of nucleotides required on both sides of the TE. Default: 30
    :param terminal_win: int, number of columns at each TE end used to check if a sequence contains this end. Default: 30
    :param min_terminal_identity: float (0-1), minimum proportion of the terminal columns in which a sequence has to
    be identical with the most abundant nucleotide. Default: 0.6
    :return: list of tuples (sequence id, left flank, TE sequence, right flank). Sequences are upper case without gaps
    """
    te = msa[:, start:end]

    # Find the columns of the TE that are supported by enough sequences
    counts = nucleotide_counts(te)
    gap_fraction = (te == b'-').sum(axis=0) / te.shape[0]
    solid_columns = np.flatnonzero((counts.sum(axis=0) >= 5) & (gap_fraction <= 0.8))
    if len(solid_columns) < 2 * terminal_win:
        return []
    majority = np.array(NUCLEOTIDES)[counts.argmax(axis=0)]

    left_columns = solid_columns[:terminal_win]
    right_columns = solid_columns[-terminal_win:]
    min_identical = min_terminal_identity * terminal_win

    flanks = []
    for seq_id, row in zip(ids, msa):
        left_identical = (row[start:end][left_columns] == majority[left_columns]).sum()
        right_identical = (row[start:end][right_columns] == majority[right_columns]).sum()
        if left_identical < min_identical or right_identical < min_identical:
            continue

        left_flank = row[:start][row[:start] != b'-'].tobytes().decode()
        te_seq = row[start:end][row[start:end] != b'-'].tobytes().decode()
        right_flank = row[end:][row[end:] != b'-'].tobytes().decode()
        if len(left_flank) < flank_len or len(right_flank) < flank_len:
            continue
        flanks.append((seq_id, left_flank, te_seq, right_flank))

    return flanks


def remove_duplicated_loci(flanks, flank_len=30, max_offset=10):
    """
    Keep one sequence for each genomic locus. The same TE copy is present several times in the MSA when several
    BLAST hits of this copy were extended. These sequences share the flanking regions on both sides.

    :param flanks: list of tuples (sequence id, left flank, TE sequence, right flank)
    :param flank_len: int, number of nucleotides next to the TE used for the comparison. Default: 30
    :param max_offset: int, the flanks of two sequences from one locus can be shifted by this number. Default: 10
    :return: list of tuples with one sequence for each locus
    """
    unique_flanks = []
    for flank in flanks:
        left_core = flank[1][-flank_len:-max_offset]
        right_core = flank[3][max_offset:flank_len]
        duplicated = any(
            left_core in kept[1][-(flank_len + max_offset) :]
            and right_core in kept[3][: flank_len + max_offset]
            for kept in unique_flanks
        )
        if not duplicated:
            unique_flanks.append(flank)
    return unique_flanks


def scan_flanks(flanks, max_shift=10, min_len=2, max_len=20):
    """
    Test if the left and right flank of the same TE copy share a short sequence directly next to the TE.
    TSDs differ between the copies of a TE family. The number of copies whose own flanks match is therefore compared
    with the match probability between the left flank of one copy and the right flank of another copy.
    The test is repeated after moving both boundaries by up to max_shift nucleotides.

    :param flanks: list of tuples (sequence id, left flank, TE sequence, right flank)
    :param max_shift: int, maximum number of nucleotides each boundary is moved in both directions. Default: 10
    :param min_len: int, minimum TSD length. Default: 2
    :param max_len: int, maximum TSD length. Default: 20
    :return: dictionary that describes the best supported boundary shift, None if there are fewer than 5 copies
    """
    copy_n = len(flanks)
    if copy_n < 5:
        return None

    shifts = range(-max_shift, max_shift + 1)
    lengths = range(min_len, max_len + 1)
    window = max_shift + max_len

    # Join flank and TE. A positive shift moves the boundary to the right
    sequences = [left + te + right for _, left, te, right in flanks]
    left_ends = [len(left) for _, left, _, _ in flanks]
    right_starts = [len(left) + len(te) for _, left, te, _ in flanks]

    # Probability that two random nucleotides of the flanking regions are identical
    flank_nucleotides = Counter(
        ''.join(left[-window:] + right[:window] for _, left, _, right in flanks)
    )
    acgt_n = sum(flank_nucleotides[nucleotide] for nucleotide in 'ACGT')
    background = sum((flank_nucleotides[nucleotide] / acgt_n) ** 2 for nucleotide in 'ACGT')

    def clean_kmer(kmer, length):
        if len(kmer) != length or set(kmer) - set('ACGT') or is_low_complexity(kmer):
            return None
        return kmer

    # left_kmers[shift][length] contains one k-mer for each copy, the same for right_kmers
    left_kmers = {
        shift: {
            length: [
                clean_kmer(seq[max(pos + shift - length, 0) : pos + shift], length)
                for seq, pos in zip(sequences, left_ends)
            ]
            for length in lengths
        }
        for shift in shifts
    }
    right_kmers = {
        shift: {
            length: [
                clean_kmer(seq[pos + shift : pos + shift + length], length)
                for seq, pos in zip(sequences, right_starts)
            ]
            for length in lengths
        }
        for shift in shifts
    }
    left_counts = {
        shift: {length: Counter(k for k in left_kmers[shift][length] if k) for length in lengths}
        for shift in shifts
    }
    right_counts = {
        shift: {length: Counter(k for k in right_kmers[shift][length] if k) for length in lengths}
        for shift in shifts
    }

    best = None
    for left_shift in shifts:
        for right_shift in shifts:
            # Longest TSD of each copy, 0 if the flanks of the copy do not match
            copy_tsd_len = [0] * copy_n
            match_probability = {}
            for length in lengths:
                left = left_kmers[left_shift][length]
                right = right_kmers[right_shift][length]
                own_matches = 0
                for i in range(copy_n):
                    if left[i] is not None and left[i] == right[i]:
                        own_matches += 1
                        copy_tsd_len[i] = length

                # Probability that the left flank of one copy matches the right flank of another copy
                left_count = left_counts[left_shift][length]
                right_count = right_counts[right_shift][length]
                all_matches = sum(n * right_count[kmer] for kmer, n in left_count.items())
                match_probability[length] = max(
                    (all_matches - own_matches) / (copy_n * (copy_n - 1)),
                    background**length,
                )

            # Test each minimum TSD length. Start with the longest
            copies_per_len = Counter(copy_tsd_len)
            support = 0
            probability = 0
            for min_tsd_len in reversed(lengths):
                support += copies_per_len[min_tsd_len]
                probability += match_probability[min_tsd_len]
                if support == 0:
                    continue
                log10_p = log10_binomial_tail(support, copy_n, probability)
                shift_size = abs(left_shift) + abs(right_shift)
                if best is None or (log10_p, shift_size) < (best['log10_p'], best['shift_size']):
                    best = {
                        'log10_p': log10_p,
                        'shift_size': shift_size,
                        'left_shift': left_shift,
                        'right_shift': right_shift,
                        'support': support,
                        'min_tsd_len': min_tsd_len,
                        'copy_tsd_len': copy_tsd_len,
                    }

    if best is None:
        return {'log10_p': 0.0, 'support': 0, 'copy_n': copy_n, 'tsd_len': 0, 'left_shift': 0, 'right_shift': 0,
                'tsds': []}

    # Correct the p value for the number of tested boundary shifts and minimum TSD lengths
    best['log10_p'] = min(best['log10_p'] + math.log10(len(shifts) ** 2 * len(lengths)), 0.0)
    best['copy_n'] = copy_n
    supporting = [
        (i, length) for i, length in enumerate(best['copy_tsd_len']) if length >= best['min_tsd_len']
    ]
    best['tsd_len'] = Counter(length for _, length in supporting).most_common(1)[0][0]
    best['tsds'] = [
        (flanks[i][0], sequences[i][right_starts[i] + best['right_shift'] :][:length])
        for i, length in supporting
    ]
    return best


def find_tsd(msa_file, start, end, anchor=True, max_shift=10, min_len=2, max_len=20, max_p_value=0.001):
    """
    Look for target site duplications (TSDs) next to the TE boundaries of an MSA.

    :param msa_file: str, path to the MSA that contains the TE and its flanking regions
    :param start: int, first MSA column of the TE (0-based)
    :param end: int, MSA column after the last column of the TE
    :param anchor: bool, move the boundaries to the edges of the conserved block before the TSD search. Default: True
    :param max_shift: int, maximum number of nucleotides each boundary is moved to look for TSDs. Default: 10
    :param min_len: int, minimum TSD length. Default: 2
    :param max_len: int, maximum TSD length. Default: 20
    :param max_p_value: float, TSDs are reported when the corrected p value is smaller than this number. Default: 0.001
    :return: dictionary with the keys of scan_flanks and
    'tested': False if there are fewer than 5 complete copies from different loci
    'found': True if TSDs are found
    'start', 'end': MSA columns of the boundaries used for the TSD search
    'start_moved', 'end_moved': number of consensus positions between the given boundaries and the boundaries used
    for the TSD search. Positive numbers point to the right
    'start_edge', 'end_edge': True if the conserved block of the MSA has a clear edge at this boundary
    'complete_sequences': number of sequences that contain both TE ends
    'copy_n': number of different loci among these sequences
    """
    ids, msa = read_msa(msa_file)
    used_start, used_end, start_edge, end_edge = start, end, False, False
    if anchor:
        used_start, used_end, start_edge, end_edge = anchor_boundaries(msa, start, end)

    flank_len = max_shift + max_len
    complete_flanks = collect_flanks(ids, msa, used_start, used_end, flank_len=flank_len)
    flanks = remove_duplicated_loci(complete_flanks, flank_len=flank_len, max_offset=max_shift)
    result = scan_flanks(flanks, max_shift=max_shift, min_len=min_len, max_len=max_len) or {}

    result['tested'] = 'log10_p' in result
    result['found'] = (
        result['tested']
        and result['log10_p'] < math.log10(max_p_value)
        and result['support'] >= 3
    )
    result['start'] = used_start
    result['end'] = used_end
    result['start_edge'] = start_edge
    result['end_edge'] = end_edge
    in_consensus = consensus_columns(msa)
    result['start_moved'] = int(in_consensus[start:used_start].sum() - in_consensus[used_start:start].sum())
    result['end_moved'] = int(in_consensus[end:used_end].sum() - in_consensus[used_end:end].sum())
    result['complete_sequences'] = len(complete_flanks)
    result['copy_n'] = len(flanks)
    return result


def format_tsd(result, reverse_complement=False):
    """
    Convert the result of find_tsd to the text that is written to the summary table.

    :param result: dictionary from find_tsd, None if the TSD search failed
    :param reverse_complement: bool, True if the MSA was reverse complemented after the TSD search. Default: False
    :return: str, 'NaN' if TSDs could not be tested and 'False' if no TSDs are found. Otherwise the TSD length, the
    number of copies with TSDs out of the tested copies and the number of nucleotides the left and right boundary
    have to be moved to be next to the TSDs, like '5bp;22/29;+0/-3'. Positive numbers point to the right
    """
    if result is None or not result['tested']:
        return 'NaN'
    if not result['found']:
        return 'False'

    left_move = result['start_moved'] + result['left_shift']
    right_move = result['end_moved'] + result['right_shift']
    if reverse_complement:
        left_move, right_move = -right_move, -left_move
    return f"{result['tsd_len']}bp;{result['support']}/{result['copy_n']};{left_move:+d}/{right_move:+d}"
