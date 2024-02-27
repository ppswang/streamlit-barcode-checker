import streamlit as st
import numpy as np
import pandas as pd

from io import StringIO
import matplotlib.pyplot as plt

import seaborn as sns


def hamming(s1, s2):
    d = 0
    s1 = s1.upper()
    s2 = s2.upper()
    for n in range( min(len(s1), len(s2) )):
        if s1[n] == "N" or s2[n] == "N":
            continue
        elif s1[n] != s2[n]:
            d += 1
    return d

def barcode_distance(b1, b2):
    if len(b1) == 1:
        return hamming(b1[0], b2[0])
    else:
        return hamming(b1[0], b2[0]) + hamming(b1[1], b2[1])


MIN_DISTANCE = 3

st.title("Barcode distance checker")
uploaded_file = st.file_uploader("Upload a samplesheet")
if uploaded_file is not None:
    filecontent = uploaded_file.read().decode("utf-8")

    # find line to start
    data_header_line = -1
    line_ct = 0
    for line in filecontent.split("\n"):
        if line.startswith("[Data]"):
            data_header_line = line_ct
        line_ct += 1

    if data_header_line < 0:
        st.write("Uploaded samplesheet does not contain a [Data] section")
    else:
        df = pd.read_csv(StringIO(filecontent), skiprows=data_header_line+1)

        colnames = df.columns
        # looking for Sample_ID, index, index2
        sample_col = -1
        index_col  = -1
        index2_col = -1
        for cidx, col in enumerate(df.columns):
            if col == "Sample_ID":
                sample_col = cidx
                df[col] = df[col].fillna("")
            elif col == "index":
                index_col = cidx
                df[col] = df[col].fillna("")
            elif col == "index2":
                index2_col = cidx
                df[col] = df[col].fillna("")

        "'Sample_ID' column: ", sample_col > -1
        "'index' column:     ", index_col > -1
        "'index2' column:    ", index2_col > -1

        st.title("Input Samplesheet [Data]")
        st.write(df)

        samples = []
        idx1    = []
        idx2    = []
        for rid, row in df.iterrows():
            row_valid = True
            # get sample ID
            if row["Sample_ID"] == "" or sample_col < 0:
                test_sample_name = "SAMPLE_%d" % rid
            else:
                test_sample_name = row["Sample_ID"]
            while test_sample_name in samples:
                test_sample_name += "_"
            # get index sequence
            index_seq = ""
            if row["index"] == "":
                row_valid == False
            else:
                index_seq = row["index"]
            # get index2 sequence
            index2_seq = ""
            if index2_col >= 0:
                if row["index2"] == "":
                    row_valid = False
                else:
                    index2_seq = row["index2"]

            if row_valid == True:
                samples.append(test_sample_name)
                idx1.append(index_seq)
                idx2.append(index2_seq)

        pairwise_distances = {}
        b1, b2 = [], []
        max_d = 0
        problem_pairs = []
        rowdata = {}
        for n1 in range(len(samples)):
            s1 = samples[n1]
            if index2_col < 0:
                b1 = [idx1[n1]]
            else:
                b1 = [idx1[n1], idx2[n1]]
            rowdata[s1] = []
            for n2 in range(len(samples)):
                # if n2 <= n1:
                #     continue
                s2 = samples[n2]
                if index2_col < 0:
                    b2 = [idx1[n2]]
                else:
                    b2 = [idx1[n2], idx2[n2]]

                if n1 == n2:
                    d = 0
                else:
                    d = barcode_distance(b1, b2)

                if d > max_d:
                    max_d = d
                if d < MIN_DISTANCE and n1 != n2:
                    problem_pairs.append( (n1, n2) )

                pairwise_distances[(s1, s2)] = d
                pairwise_distances[(s2, s1)] = d
                rowdata[s1].append(d)

        dfd = pd.DataFrame(rowdata)
        dfd["Sample"] = samples
        dfd = dfd.set_index("Sample", drop=True)

        st.title("Barcode distances")
        dfd


        fig_w = max(15, len(samples)*0.2+2)
        fig_h = max(13, len(samples)*0.2)
        plt.rcParams["figure.figsize"] = (fig_w, fig_h)

        c1 = "red"
        c2 = "#DDDDFF"
        c3 = "#DDFFDD"
        colors = []
        for n in range(max_d):
            if n < MIN_DISTANCE:
                colors.append(c1)
            elif n < MIN_DISTANCE + 2:
                colors.append(c2)
            else:
                colors.append(c3)
        sns.heatmap(dfd,
                    cmap=colors,
                    annot=True,
                    linewidth=0.5,
                    cbar=False,
                    linecolor="black",
                   )
        fig = plt.gcf()
        st.pyplot(fig)

        if len(problem_pairs) > 0:
            st.title("Conflicts")
            st.write("These pairs have conflicts")
            for i in range(len(problem_pairs)):
                n1, n2 = problem_pairs[i]
                if n1 > n2:
                    continue
                b1 = idx1[n1]
                b2 = idx1[n2]
                if index2_col >= 0:
                    b1 += "+" + idx2[n1]
                    b2 += "+" + idx2[n2]
                s1 = samples[n1]
                s2 = samples[n2]
                st.write("Pair %d" % (i+1))
                st.write("%s [%s] vs" % (s1, b1))
                st.write("%s [%s] (d = %d) " % (s2, b2, pairwise_distances[(s1, s2)]))
                st.write("-------------------------")

#
