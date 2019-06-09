# test_size = 0.25 --> pour choisir params
# puis entrainement ac 0.01
# verifier que taskname posos = ok
#
# test_size = 0.25 --> pour choisir params
# puis entrainement ac 0.01
#
# verifier que taskname posos = ok
#
# batch size: [16,32]  --> essayer aussi ac 8
# learning rate: [2e-5, 3e-5, 5e-5]  --> essayer aussi ac 1e-5
# nb of epochs [2,3,4] essayer aussi 5
#
#
# noter le temps a chaque fois
#
# rajouter une epoch 3 a 4 a augmente de 1 a 3 %
#
# processors = {
#      "cola": ColaProcessor,
#      "mnli": MnliProcessor,
#      "mrpc": MrpcProcessor,
#      "xnli": XnliProcessor,
#      "selfsim": SelfProcessor #添加自己的processor
#   }
#
# max seq length 64 ?


with open('b_training_script_test_processors', 'a+') as f:
    i = 18
    for train_batch_size in [16]:
        for learning_rate in [2e-5]: # , 2e-5, 3e-5, 4e-5, 5e-5]:
            for epochs in [1]: # 2, 3, 4]:
                for task in ['cola', 'mnli', 'mrpc', 'xnli']:
                    i += 1


                    train_script = """
    # output = {}
    time
    
    python run_classifier.py \\
    --task_name={} \\
    --do_train=true \\
    --do_eval=true \\
    --do_predict=true \\
    --data_dir=./data/ \\
    --vocab_file=./multi_cased_L-12_H-768_A-12/vocab.txt \\
    --bert_config_file=./multi_cased_L-12_H-768_A-12/bert_config.json \\
    --init_checkpoint=./multi_cased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 \\
    --train_batch_size={} \\
    --learning_rate={} \\
    --num_train_epochs={} \\
    --output_dir=./output_{}/ \\
    --do_lower_case=False
    
    time
    # ________________
                """.format(i, task, train_batch_size, learning_rate, epochs, i)

                    print(train_script, file=f)



#
# batch size: [16,32]  --> essayer aussi ac 8
# learning rate: [2e-5, 3e-5, 5e-5]  --> essayer aussi ac 1e-5
# nb of epochs [2,3,4] essayer aussi 5
#
# python run_classifier.py \
# --task_name=posos \
# --do_train=true \
# --do_eval=true \
# --do_predict=true \
# --data_dir=./data/ \
# --vocab_file=./multi_cased_L-12_H-768_A-12/vocab.txt \
# --bert_config_file=./multi_cased_L-12_H-768_A-12/bert_config.json \
# --init_checkpoint=./multi_cased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 \
# --train_batch_size=32 \
# --learning_rate=5e-6 \
# --num_train_epochs=8.0 \
# --output_dir=./output_4/ \
# --do_lower_case=False
#
#
# python run_classifier.py \
# --task_name=cola \
# --do_predict=true \
# --data_dir=./data \
# --vocab_file=./multi_cased_L-12_H-768_A-12/vocab.txt \
# --bert_config_file=./multi_cased_L-12_H-768_A-12/bert_config.json \
# --init_checkpoint=./output_9/model.ckpt-2862 \
# --max_seq_length=128 \
# --output_dir=./output_9/

