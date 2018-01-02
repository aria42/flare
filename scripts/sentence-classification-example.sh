#!/usr/bin/env bash

# shuffling lines (or filenames)
myshuf() {
  perl -MList::Util=shuffle -e 'print shuffle(<>);'
}

# whitespace tokenization
normalize_text() {
  sed -e 's/<br \/>/ /g' | \
    -e "s/$/ /" -e "s/^/ /" \
    -e 's/\([^A-Za-z0-9]\) / \1 /g' -e 's/ \([^A-Za-z0-9]\)/ \1 /g' \
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
if [ ! -f "${DATADIR}/glove.6B.300d.txt" ]
then
  wget -c "http://nlp.stanford.edu/data/glove.6B.zip" \
    -O "${DATADIR}/glove.6B.zip"
  unzip "${DATADIR}/glove.6B.zip" -d "${DATADIR}/"
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

# run the sentence classification
set -x
lein with-profile main-sentclass run \
  --model-type bilstm --num-data 8000 \
  --embed-file "${DATADIR}/glove.6B.300d.txt" --emb-size 300
set +x
