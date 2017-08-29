export PYTHONIOENCODING=UTF-8
export DATA_PATH="$(pwd)/data_giga"

export VOCAB_SOURCE=${DATA_PATH}/vocab_article_65k
export VOCAB_TARGET=${DATA_PATH}/vocab_title_65k
export TRAIN_SOURCES=${DATA_PATH}/train.article.txt
export TRAIN_TARGETS=${DATA_PATH}/train.title.txt
export DEV_SOURCES=${DATA_PATH}/valid.article.filter.shuffle.2000.txt
export DEV_TARGETS=${DATA_PATH}/valid.title.filter.shuffle.2000.txt
export TEST_SOURCES=${DATA_PATH}/input.txt
export TEST_TARGETS=${DATA_PATH}/task1_ref0.txt
export TOPIC_MODEL="$(pwd)"/giga_lda_model0716_

export TRAIN_STEPS=1000000

export MODEL_DIR="$(pwd)/sum_conv_seq2seq_topic_rl"
mkdir -p $MODEL_DIR

python -m bin.train \
  --config_paths="
      ./example_configs/conv_seq2seq_pretrain.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_sp.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET
      topic_model: $TOPIC_MODEL" \
  --input_pipeline_train "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipelineFairseq
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 128 \
  --eval_every_n_steps 500 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR \
  --save_checkpoints_steps 500 \
  --keep_checkpoint_max 1000
