export PYTHONIOENCODING=UTF-8

DATA_PATH="$(pwd)/data_giga"
MODEL_DIR="$(pwd)/sum_conv_seq2seq_topic_rl"
###DATA_PATH=$1
###TEST_SOURCES=$2
###MODEL_DIR=$3

export VOCAB_SOURCE=${DATA_PATH}/vocab_article_50k
export VOCAB_TARGET=${DATA_PATH}/vocab_title_50k
export TRAIN_SOURCES=${DATA_PATH}/train.article.txt
export TRAIN_TARGETS=${DATA_PATH}/train.title.txt
export DEV_SOURCES=${DATA_PATH}/valid.article.filter_small.txt
export DEV_TARGETS=${DATA_PATH}/valid.title.filter_small.txt
export TEST_SOURCES=${DATA_PATH}/input.txt
export TEST_TARGETS=${DATA_PATH}/task1_ref0.txt
export TOPIC_MODEL="$(pwd)"/giga_lda_model0716_

##export PRED_DIR=${DATA_PATH}/summary
export PRED_DIR=${MODEL_DIR}/pred
export GREEDY_DIR=${DATA_PATH}/summary/greedy
export BEAM_DIR=${DATA_PATH}/summary/beam
mkdir -p ${PRED_DIR}
mkdir -p ${GREEDY_DIR}
mkdir -p ${BEAM_DIR}


echo "Greedy search..."
###with greedy search
python -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 1 
    decoder.class: seq2seq.decoders.ConvDecoderFairseq" \
  --input_pipeline "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TEST_SOURCES" \
  > ${PRED_DIR}/summaryA.txt
rm ${GREEDY_DIR}/summaryA.txt
cp ${PRED_DIR}/summaryA.txt ${GREEDY_DIR}/summaryA.txt
rm ${PRED_DIR}/summaryA.txt

: <<END
echo "Beam search..."
###with beam search
python -m bin.infer \
  --tasks "
    - class: DecodeText
    - class: DumpBeams
      params:
        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 5 
    decoder.class: seq2seq.decoders.ConvDecoderFairseqBS" \
  --input_pipeline "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TEST_SOURCES" \
  | sed 's/@@ //g'> ${PRED_DIR}/summaryA.txt
cp ${PRED_DIR}/summaryA.txt ${BEAM_DIR}/summaryA.txt
rm ${PRED_DIR}/summaryA.txt
END

echo "Greedy result:" 
python seq2seq/metrics/pythonrouge/rouge_scorer.py -ref_dir $DATA_PATH/reference/origin -sum_dir ${GREEDY_DIR}
###echo "Beam result:"
###python seq2seq/metrics/pythonrouge/rouge_scorer.py -ref_dir $DATA_PATH/reference/origin -sum_dir ${BEAM_DIR}


: <<END
(tensorflow) nlp@nlp-workstation:/nlp/lilianwang/conv_seq2seq_master$ python seq2seq/metrics/pythonrouge/rouge_scorer.py -ref_dir ./sum_conv_seq2seq_rl/reference -sum_dir ./sum_conv_seq2seq_rl/summary/greedy/
{'ROUGE-1': 0.82474, 'ROUGE-2': 0.69474, 'ROUGE-L': 0.82474, 'ROUGE-SU4': 0.72}
(tensorflow) nlp@nlp-workstation:/nlp/lilianwang/conv_seq2seq_master$ python seq2seq/metrics/pythonrouge/rouge_scorer.py -ref_dir ./sum_conv_seq2seq_rl/reference -sum_dir ./sum_conv_seq2seq_rl/summary/beam/
{'ROUGE-1': 0.76288, 'ROUGE-2': 0.61053, 'ROUGE-L': 0.76288, 'ROUGE-SU4': 0.62545}
"""
END

## | sed 's/@@ //g'
