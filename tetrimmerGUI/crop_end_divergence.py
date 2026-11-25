import os

import click
import numpy as np
import pandas as pd
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


class CropEnd:
    """
    Crop each single sequence end of the MSA by nucleotide divergence.
    """

    def __init__(self, input_file, threshold=16, window_size=20):
        """
        :param input_file: str, path to the multiple sequence alignment
        :param threshold: default 16, nucleotide number inside the checking window whose proportion should be greater than 80%
        :param window_size: default 20, checking window size to define start and end position
        """
        self.input_file = input_file
        self.alignment = AlignIO.read(self.input_file, 'fasta')
        self.threshold = threshold
        self.window_size = window_size
        # Define a dictionary to hold start and end positions
        self.position_dict = {record.id: [0, 0] for record in self.alignment}
        # Define an empty dataframe to store proportion information
        self.df = None
        self.cropped_alignment = []
        self.pro_calculation()
        self.find_positions()

    def pro_calculation(self):
        """
        :function pro_calculation: calculate nucleotide proportion in each column of the MSA
        :return: a data frame containing all sequence names and the respective nucleotide proportion information for each position
        """
        # Store the alignment as a matrix. Each row represents a sequence and each column represents a position
        alignment_matrix = np.array(
            [
                np.frombuffer(str(record.seq).encode(), dtype='S1')
                for record in self.alignment
            ]
        )
        nucleotides = [b'a', b'c', b'g', b't']

        # Count the number of each nucleotide in every column
        counts = np.array(
            [
                (
                    (alignment_matrix == nucleotide)
                    | (alignment_matrix == nucleotide.upper())
                ).sum(axis=0)
                for nucleotide in nucleotides
            ]
        )
        total = counts.sum(axis=0)

        # Calculate the proportion of each nucleotide for every column
        # Ignore columns less 5 nucleotides
        proportions = np.zeros(counts.shape)
        np.divide(counts, total, out=proportions, where=total >= 5)

        # Add the proportion of the nucleotide at each position to each sequence
        # When there is a gap, use NaN replace proportion
        seq_proportions = np.full(alignment_matrix.shape, np.nan)
        for nucleotide, nucleotide_proportions in zip(nucleotides, proportions):
            seq_proportions = np.where(
                alignment_matrix == nucleotide, nucleotide_proportions, seq_proportions
            )

        # Each row of the DataFrame represents a sequence and the columns represent positions
        self.df = pd.DataFrame(
            seq_proportions,
            index=[record.id for record in self.alignment],
            columns=range(1, self.alignment.get_alignment_length() + 1),
        )
        # Convert to two decimal numbers
        self.df = self.df.round(2)

    def candidate_windows(self, cumulative):
        """
        Find the windows whose sum of proportions is close to or greater than the threshold.

        :param cumulative: numpy array, cumulative sum of the proportions of one sequence in hundredths, starts with 0
        :return: numpy array containing the start positions of the windows in ascending order
        """
        window_number = max(len(cumulative) - self.window_size, 0)
        window_sums = (
            cumulative[self.window_size : self.window_size + window_number]
            - cumulative[:window_number]
        )
        # Keep a margin of one hundredth. The window sum itself is compared with the threshold afterwards
        return np.flatnonzero(window_sums > self.threshold * 100 - 1)

    def find_positions(self):
        """
        This function will define the start and end position for each sequence
            by nucleotide proportions.

        :return: a dictionary that contains sequence name, start, and end positions
        """
        # Loop over the DataFrame's rows
        for index, row in self.df.iterrows():
            # Gaps do not contribute to the sum of a window
            proportions = row.fillna(0).to_numpy()
            # The proportions have two decimals, their cumulative sum is exact when counted in hundredths
            cumulative = np.concatenate(
                ([0], np.rint(proportions * 100).astype(int).cumsum())
            )
            candidates = self.candidate_windows(cumulative)

            # Find start position
            for i in candidates:
                window = proportions[i : i + self.window_size]
                if window.sum() > self.threshold:
                    self.position_dict[index][0] = int(i)
                    break
            # Find end position
            for i in candidates[::-1]:
                window = proportions[i : i + self.window_size]
                if window.sum() > self.threshold:
                    self.position_dict[index][1] = int(i) + self.window_size
                    break

    def crop_alignment(self):
        # Create a new list to hold the cropped sequences
        # Loop through each sequence in the alignment
        for record in self.alignment:
            # Create a new string with the cropped sequence
            cropped_seq = (
                '-' * self.position_dict[record.id][0]
                + str(
                    record.seq[
                        self.position_dict[record.id][0] : self.position_dict[
                            record.id
                        ][1]
                    ]
                )
                + '-' * (len(record.seq) - self.position_dict[record.id][1])
            )
            # Create a new SeqRecord with the cropped sequence and add it to the list
            self.cropped_alignment.append(
                SeqRecord(Seq(cropped_seq), id=record.id, description='')
            )
        # Convert the list of SeqRecords into a MultipleSeqAlignment
        self.cropped_alignment = MultipleSeqAlignment(self.cropped_alignment)

        return self.cropped_alignment

    def average_proportion_per_column(self):
        """
        Calculate the average proportion for each column across all sequences.
        """
        # Ensure the DataFrame is available
        if not hasattr(self, 'df'):
            raise ValueError(
                'The DataFrame has not been created yet. Please run pro_calculation() first.'
            )

        # Calculate the average for each column
        average_proportions = self.df.mean()

        # Calculate the overall mean of the column averages
        overall_average = average_proportions.mean()

        return average_proportions, overall_average

    def write_to_file(self, output_dir):
        output_file = os.path.join(
            output_dir, f'{os.path.basename(self.input_file)}_ce.fa'
        )
        with open(output_file, 'w') as f:
            AlignIO.write(self.cropped_alignment, f, 'fasta')
        return output_file


def crop_end_div(input_file, output_file, threshold, window_size):
    if os.path.isfile(input_file):
        crop_end_thr = threshold * window_size
        crop_end_div_object = CropEnd(
            input_file, threshold=crop_end_thr, window_size=window_size
        )
        crop_alignment = crop_end_div_object.crop_alignment()
        with open(output_file, 'w') as f:
            AlignIO.write(crop_alignment, f, 'fasta')
    else:
        raise FileNotFoundError(f"The file '{input_file}' does not exist.")


@click.command()
@click.option(
    '--input_file',
    '-i',
    required='True',
    type=str,
    help='Multiple sequence alignment FASTA file path',
)
@click.option('--output_file', '-o', required='True', type=str, help='Output file')
@click.option(
    '--threshold',
    '-thr',
    default=0.8,
    type=float,
    help='Higher number means more stringent cropping. Range: 0-1',
)
@click.option(
    '--window_size',
    '-ws',
    default=20,
    type=int,
    help='Window size used for cropping end',
)
def crop_end_div_click(input_file, output_file, threshold, window_size):
    crop_end_div(input_file, output_file, threshold, window_size)


if __name__ == '__main__':
    crop_end_div_click()
