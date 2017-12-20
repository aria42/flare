#!/usr/bin/env bash

# shuffling lines (or filenames)
myshuf() {
  perl -MList::Util=shuffle -e 'print shuffle(<>);'
}

# PTB tokenization
normalize_text() {
  sed -e 's/<br \/>/ /g' | \
    sed -e 's/^"/`` /g' -e 's/\([ ([{<]\)"/\1 `` /g' \
        -e "s/\([^ \.]\)\./\1 ./g" -e "s/\.\.\./ ... /g" \
        -e "s/[,;:@#$%&]/ & /g" \
        -e "s/\([^.]\)\([.]\)\([])}>\"']*\)[ 	]*$/\1 \2\3 /g" \
        -e "s/[?!]/ & /g" -e "s/[][(){}<>]/ & /g" -e "s/--/ -- /g" \
        -e "s=/= / =g" -e "s/$/ /" -e "s/^/ /" -e "s/\"/ '' /g" \
        -e "s/\([^']\)' /\1 ' /g" -e "s/'\([sSmMdD]\) / '\1 /g" \
        -e "s/'ll / 'll /g" -e "s/'re / 're /g" -e "s/'ve / 've /g" \
        -e "s/n't / n't /g" -e "s/'LL / 'LL /g" -e "s/'RE / 'RE /g" \
        -e "s/'VE / 'VE /g" -e "s/N'T / N'T /g" \
        -e "s/ \([Cc]\)annot / \1an not /g" \
        -e "s/ \([Dd]\)'ye / \1' ye /g" \
        -e "s/ \([Gg]\)imme / \1im me /g" \
        -e "s/ \([Gg]\)onna / \1on na /g" \
        -e "s/ \([Gg]\)otta / \1ot ta /g" \
        -e "s/ \([Ll]\)emme / \1em me /g" \
        -e "s/ \([Mm]\)ore'n / \1ore 'n /g" \
        -e "s/ '\([Tt]\)is / '\1 is /g" \
        -e "s/ '\([Tt]\)was / '\1 was /g" \
        -e "s/ \([Ww]\)anna / \1an na /g" \
        -e "s/  */ /g" -e "s/^ *//g" -e "s/|/ /g" | \
        tr '[:upper:]' '[:lower:]' | tr -s " " | tr -cd '[:print:]\n'
}

DATADIR=data
NUM_EXAMPLES=5000
SCRIPT_DIR=`dirname $0`

# create the data directory
mkdir -p "${DATADIR}"

# download the IMDB sentiment classification data set
set -x
if [ ! -f "${DATADIR}/aclImdb_v1.tar.gz" ]
then
  wget -c "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz" \
    -O "${DATADIR}/aclImdb_v1.tar.gz"
  tar -xzvf "${DATADIR}/aclImdb_v1.tar.gz" -C "${DATADIR}"
fi
set +x

# download the GloVe word vectors
set -x
if [ ! -f "${DATADIR}/glove.42B.300d.txt" ]
then
  wget -c "http://nlp.stanford.edu/data/glove.42B.300d.zip" \
    -O "${DATADIR}/glove.42B.300d.zip"
  unzip "${DATADIR}/glove.42B.300d.zip" -d "${DATADIR}/"
fi
set +x

IMDB_DIR="${DATADIR}/aclImdb"

# create the training file by randomly sampling 5K examples from training
# positive and negative examples and finally output a shuffled set of examples
# to sentiment-train10k.txt
IMDB_TRAIN_DIR="${IMDB_DIR}/train"
if [ ! -f "${DATADIR}/sentiment-train10k.txt" ]
then
  for f in `ls ${IMDB_TRAIN_DIR}/neg/ | myshuf`; do
    cat "${IMDB_TRAIN_DIR}/neg/$f" | normalize_text | \
      sed -e 's/^/0 /g' >> "${DATADIR}/sentiment-train-negative.txt"
  done
  for f in `ls ${IMDB_TRAIN_DIR}/pos/ | myshuf`; do
    cat "${IMDB_TRAIN_DIR}/pos/$f" | normalize_text | \
      sed -e 's/^/1 /g' >> "${DATADIR}/sentiment-train-positive.txt"
  done

  set -x
  head -n ${NUM_EXAMPLES} "${DATADIR}/sentiment-train-negative.txt" > \
    "${DATADIR}/sentiment-train10k-negative.txt"
  head -n ${NUM_EXAMPLES} "${DATADIR}/sentiment-train-positive.txt" > \
    "${DATADIR}/sentiment-train10k-positive.txt"
  cat "${DATADIR}/sentiment-train10k-negative.txt" \
    "${DATADIR}/sentiment-train10k-positive.txt" | myshuf > \
    "${DATADIR}/sentiment-train10k.txt"
  set +x
fi

# same procedure as above
IMDB_TEST_DIR="${IMDB_DIR}/test"
if [ ! -f "${DATADIR}/sentiment-test10k.txt" ]
then
  for f in `ls ${IMDB_TEST_DIR}/neg/ | myshuf`; do
    cat "${IMDB_TEST_DIR}/neg/$f" | normalize_text | \
      sed -e 's/^/0 /g' >> "${DATADIR}/sentiment-test-negative.txt"
  done
  for f in `ls ${IMDB_TEST_DIR}/pos/ | myshuf`; do
    cat "${IMDB_TEST_DIR}/pos/$f" | normalize_text | \
      sed -e 's/^/1 /g' >> "${DATADIR}/sentiment-test-positive.txt"
  done

  set -x
  head -n ${NUM_EXAMPLES} "${DATADIR}/sentiment-test-negative.txt" > \
    "${DATADIR}/sentiment-test10k-negative.txt"
  head -n ${NUM_EXAMPLES} "${DATADIR}/sentiment-test-positive.txt" > \
    "${DATADIR}/sentiment-test10k-positive.txt"
  cat "${DATADIR}/sentiment-test10k-negative.txt" \
    "${DATADIR}/sentiment-test10k-positive.txt" | myshuf > \
    "${DATADIR}/sentiment-test10k.txt"
  set +x
fi

# filter the GloVe vectors (restricted to training and testing words)
set -x
if [ ! -f "${DATADIR}/small-glove.300d.txt" ]
then
  cat "${DATADIR}/glove.42B.300d.txt" | \
    "${SCRIPT_DIR}/filter-glove.pl" \
      "${DATADIR}/sentiment-train-negative.txt" \
      "${DATADIR}/sentiment-train-positive.txt" \
      "${DATADIR}/sentiment-test-negative.txt" \
      "${DATADIR}/sentiment-test-positive.txt" > \
      "${DATADIR}/small-glove.300d.txt"
fi
set +x

# run the sentence classification
set -x
lein with-profile main-sentclass run \
  --model-type cnn --emb-size 300 --num-data 10000
set +x
