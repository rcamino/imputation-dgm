#!/usr/bin/env bash

########################################################################################################################
# DOWNLOAD ENCODE AND SCALE
########################################################################################################################

echo "[DOWNLOAD AND ENCODE AND SCALE]"
python imputation_dgm/pre_processing/breast/download_and_transform.py
python imputation_dgm/pre_processing/default_credit_card/download_and_transform.py
python imputation_dgm/pre_processing/letter_recognition/download_and_transform.py
python imputation_dgm/pre_processing/online_news_popularity/download_and_transform.py
python imputation_dgm/pre_processing/spambase/download_and_transform.py

########################################################################################################################
# TRAIN AND TEST SPLIT
########################################################################################################################

echo "[TRAIN AND TEST SPLIT]"
python imputation_dgm/pre_processing/train_test_split.py data/breast/features.npy 0.9 data/breast/features-train.npy data/breast/features-test.npy --features_format=dense
python imputation_dgm/pre_processing/train_test_split.py data/default-credit-card/features.npy 0.9 data/default-credit-card/features-train.npy data/default-credit-card/features-test.npy --features_format=dense
python imputation_dgm/pre_processing/train_test_split.py data/letter-recognition/features.npy 0.9 data/letter-recognition/features-train.npy data/letter-recognition/features-test.npy --features_format=dense
python imputation_dgm/pre_processing/train_test_split.py data/online-news-popularity/features.npy 0.9 data/online-news-popularity/features-train.npy data/online-news-popularity/features-test.npy --features_format=dense
python imputation_dgm/pre_processing/train_test_split.py data/spambase/features.npy 0.9 data/spambase/features-train.npy data/spambase/features-test.npy --features_format=dense

########################################################################################################################
# GAIN
########################################################################################################################

for miss_prob in 0.2 0.5 0.8; do
    for seed in 1 2 3 4; do
        echo "[GAIN breast missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/gain/trainer.py \
            "data/breast/features-train.npy" \
            "data/breast/features-test.npy" \
            "data/breast/metadata.json" \
            "models/breast/gain/miss_prob${miss_prob}/seed${seed}/generator.torch" \
            "models/breast/gain/miss_prob${miss_prob}/seed${seed}/discriminator.torch" \
            "models/breast/gain/miss_prob${miss_prob}/seed${seed}/loss.csv" \
            "--max_seconds_without_save=300" \
            "--seed=${seed}" \
            "--data_format=dense" \
            "--start_epoch=0" \
            "--batch_size=64" \
            "--num_epochs=1" \
            "--learning_rate=0.001" \
            "--generator_hidden_sizes=30,15,30" \
            "--discriminator_hidden_sizes=30,15,30" \
            "--num_discriminator_steps=1" \
            "--num_generator_steps=1" \
            "--reconstruction_loss_weight=10" \
            "--missing_probability=${miss_prob}" \
            "--hint_probability=0.9"
    done

    for seed in 1 2 3 4; do
        echo "[GAIN default-credit-card missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/gain/trainer.py \
            "data/default-credit-card/features-train.npy" \
            "data/default-credit-card/features-test.npy" \
            "data/default-credit-card/metadata.json" \
            "models/default-credit-card/gain/miss_prob${miss_prob}/seed${seed}/generator.torch" \
            "models/default-credit-card/gain/miss_prob${miss_prob}/seed${seed}/discriminator.torch" \
            "models/default-credit-card/gain/miss_prob${miss_prob}/seed${seed}/loss.csv" \
            "--max_seconds_without_save=300" \
            "--seed=${seed}" \
            "--data_format=dense" \
            "--start_epoch=0" \
            "--batch_size=64" \
            "--num_epochs=1" \
            "--learning_rate=0.001" \
            "--generator_hidden_sizes=93,46,93" \
            "--discriminator_hidden_sizes=93,46,93" \
            "--num_discriminator_steps=1" \
            "--num_generator_steps=1" \
            "--reconstruction_loss_weight=10" \
            "--missing_probability=${miss_prob}" \
            "--hint_probability=0.9"
    done

    for seed in 1 2 3 4; do
        echo "[GAIN letter-recognition missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/gain/trainer.py \
            "data/letter-recognition/features-train.npy" \
            "data/letter-recognition/features-test.npy" \
            "data/letter-recognition/metadata.json" \
            "models/letter-recognition/gain/miss_prob${miss_prob}/seed${seed}/generator.torch" \
            "models/letter-recognition/gain/miss_prob${miss_prob}/seed${seed}/discriminator.torch" \
            "models/letter-recognition/gain/miss_prob${miss_prob}/seed${seed}/loss.csv" \
            "--max_seconds_without_save=300" \
            "--seed=${seed}" \
            "--data_format=dense" \
            "--start_epoch=0" \
            "--batch_size=64" \
            "--num_epochs=1" \
            "--learning_rate=0.001" \
            "--generator_hidden_sizes=16,8,16" \
            "--discriminator_hidden_sizes=16,8,16" \
            "--num_discriminator_steps=1" \
            "--num_generator_steps=1" \
            "--reconstruction_loss_weight=10" \
            "--missing_probability=${miss_prob}" \
            "--hint_probability=0.9"
    done

    for seed in 1 2 3 4; do
        echo "[GAIN online-news-popularity missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/gain/trainer.py \
            "data/online-news-popularity/features-train.npy" \
            "data/online-news-popularity/features-test.npy" \
            "data/online-news-popularity/metadata.json" \
            "models/online-news-popularity/gain/miss_prob${miss_prob}/seed${seed}/generator.torch" \
            "models/online-news-popularity/gain/miss_prob${miss_prob}/seed${seed}/discriminator.torch" \
            "models/online-news-popularity/gain/miss_prob${miss_prob}/seed${seed}/loss.csv" \
            "--max_seconds_without_save=300" \
            "--seed=${seed}" \
            "--data_format=dense" \
            "--start_epoch=0" \
            "--batch_size=64" \
            "--num_epochs=1" \
            "--learning_rate=0.001" \
            "--generator_hidden_sizes=60,30,60" \
            "--discriminator_hidden_sizes=60,30,60" \
            "--num_discriminator_steps=1" \
            "--num_generator_steps=1" \
            "--reconstruction_loss_weight=10" \
            "--missing_probability=${miss_prob}" \
            "--hint_probability=0.9"
    done

    for seed in 1 2 3 4; do
        echo "[GAIN spambase missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/gain/trainer.py \
            "data/spambase/features-train.npy" \
            "data/spambase/features-test.npy" \
            "data/spambase/metadata.json" \
            "models/spambase/gain/miss_prob${miss_prob}/seed${seed}/generator.torch" \
            "models/spambase/gain/miss_prob${miss_prob}/seed${seed}/discriminator.torch" \
            "models/spambase/gain/miss_prob${miss_prob}/seed${seed}/loss.csv" \
            "--max_seconds_without_save=300" \
            "--seed=${seed}" \
            "--data_format=dense" \
            "--start_epoch=0" \
            "--batch_size=64" \
            "--num_epochs=1" \
            "--learning_rate=0.001" \
            "--generator_hidden_sizes=57,28,57" \
            "--discriminator_hidden_sizes=57,28,57" \
            "--num_discriminator_steps=1" \
            "--num_generator_steps=1" \
            "--reconstruction_loss_weight=10" \
            "--missing_probability=${miss_prob}" \
            "--hint_probability=0.9"
    done
done

########################################################################################################################
# GAIN-split
########################################################################################################################

for miss_prob in 0.2 0.5 0.8; do
    for seed in 1 2 3 4; do
        echo "[GAIN-split default-credit-card missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/gain-split/trainer.py \
            "data/default-credit-card/features-train.npy" \
            "data/default-credit-card/features-test.npy" \
            "data/default-credit-card/metadata.json" \
            "models/default-credit-card/gain-split/miss_prob${miss_prob}/seed${seed}/generator.torch" \
            "models/default-credit-card/gain-split/miss_prob${miss_prob}/seed${seed}/discriminator.torch" \
            "models/default-credit-card/gain-split/miss_prob${miss_prob}/seed${seed}/loss.csv" \
            "--max_seconds_without_save=300" \
            "--seed=${seed}" \
            "--data_format=dense" \
            "--start_epoch=0" \
            "--batch_size=64" \
            "--num_epochs=1" \
            "--learning_rate=0.001" \
            "--generator_hidden_sizes=93,46,93" \
            "--discriminator_hidden_sizes=93,46,93" \
            "--num_discriminator_steps=1" \
            "--num_generator_steps=1" \
            "--reconstruction_loss_weight=10" \
            "--missing_probability=${miss_prob}" \
            "--hint_probability=0.9" \
            "--temperature=0.1"
    done

    for seed in 1 2 3 4; do
        echo "[GAIN-split online-news-popularity missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/gain-split/trainer.py \
            "data/online-news-popularity/features-train.npy" \
            "data/online-news-popularity/features-test.npy" \
            "data/online-news-popularity/metadata.json" \
            "models/online-news-popularity/gain-split/miss_prob${miss_prob}/seed${seed}/generator.torch" \
            "models/online-news-popularity/gain-split/miss_prob${miss_prob}/seed${seed}/discriminator.torch" \
            "models/online-news-popularity/gain-split/miss_prob${miss_prob}/seed${seed}/loss.csv" \
            "--max_seconds_without_save=300" \
            "--seed=${seed}" \
            "--data_format=dense" \
            "--start_epoch=0" \
            "--batch_size=64" \
            "--num_epochs=1" \
            "--learning_rate=0.001" \
            "--generator_hidden_sizes=60,30,60" \
            "--discriminator_hidden_sizes=60,30,60" \
            "--num_discriminator_steps=1" \
            "--num_generator_steps=1" \
            "--reconstruction_loss_weight=10" \
            "--missing_probability=${miss_prob}" \
            "--hint_probability=0.9" \
            "--temperature=0.1"
    done
done

########################################################################################################################
# VAE
########################################################################################################################

for miss_prob in 0.2 0.5 0.8; do
    for seed in 1 2 3 4; do
        echo "[VAE breast missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/trainer.py \
            "data/breast/features-train.npy" \
            "data/breast/features-test.npy" \
            "data/breast/metadata.json" \
            "models/breast/vae/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/breast/vae/miss_prob${miss_prob}/seed${seed}/loss.csv" \
            "--max_seconds_without_save=300" \
            "--seed=${seed}" \
            "--data_format=dense" \
            "--start_epoch=0" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=30,15" \
            "--decoder_hidden_sizes=30,15" \
            "--batch_size=64" \
            "--num_epochs=1" \
            "--l2_regularization=0" \
            "--learning_rate=0.001" \
            "--missing_probability=${miss_prob}"
    done

    for seed in 1 2 3 4; do
        echo "[VAE default-credit-card missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/trainer.py \
            "data/default-credit-card/features-train.npy" \
            "data/default-credit-card/features-test.npy" \
            "data/default-credit-card/metadata.json" \
            "models/default-credit-card/vae/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/default-credit-card/vae/miss_prob${miss_prob}/seed${seed}/loss.csv" \
            "--max_seconds_without_save=300" \
            "--seed=${seed}" \
            "--data_format=dense" \
            "--start_epoch=0" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=93,46" \
            "--decoder_hidden_sizes=93,46" \
            "--batch_size=64" \
            "--num_epochs=1" \
            "--l2_regularization=0" \
            "--learning_rate=0.001" \
            "--missing_probability=${miss_prob}"
    done

    for seed in 1 2 3 4; do
        echo "[VAE letter-recognition missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/trainer.py \
            "data/letter-recognition/features-train.npy" \
            "data/letter-recognition/features-test.npy" \
            "data/letter-recognition/metadata.json" \
            "models/letter-recognition/vae/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/letter-recognition/vae/miss_prob${miss_prob}/seed${seed}/loss.csv" \
            "--max_seconds_without_save=300" \
            "--seed=${seed}" \
            "--data_format=dense" \
            "--start_epoch=0" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=16,8" \
            "--decoder_hidden_sizes=16,8" \
            "--batch_size=64" \
            "--num_epochs=1" \
            "--l2_regularization=0" \
            "--learning_rate=0.001" \
            "--missing_probability=${miss_prob}"
    done

    for seed in 1 2 3 4; do
        echo "[VAE online-news-popularity missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/trainer.py \
            "data/online-news-popularity/features-train.npy" \
            "data/online-news-popularity/features-test.npy" \
            "data/online-news-popularity/metadata.json" \
            "models/online-news-popularity/vae/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/online-news-popularity/vae/miss_prob${miss_prob}/seed${seed}/loss.csv" \
            "--max_seconds_without_save=300" \
            "--seed=${seed}" \
            "--data_format=dense" \
            "--start_epoch=0" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=60,30" \
            "--decoder_hidden_sizes=60,30" \
            "--batch_size=64" \
            "--num_epochs=1" \
            "--l2_regularization=0" \
            "--learning_rate=0.001" \
            "--missing_probability=${miss_prob}"
    done

    for seed in 1 2 3 4; do
        echo "[VAE spambase missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/trainer.py \
            "data/spambase/features-train.npy" \
            "data/spambase/features-test.npy" \
            "data/spambase/metadata.json" \
            "models/spambase/vae/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/spambase/vae/miss_prob${miss_prob}/seed${seed}/loss.csv" \
            "--max_seconds_without_save=300" \
            "--seed=${seed}" \
            "--data_format=dense" \
            "--start_epoch=0" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=57,28" \
            "--decoder_hidden_sizes=57,28" \
            "--batch_size=64" \
            "--num_epochs=1" \
            "--l2_regularization=0" \
            "--learning_rate=0.001" \
            "--missing_probability=${miss_prob}"
    done
done

########################################################################################################################
# VAE-split
########################################################################################################################

for miss_prob in 0.2 0.5 0.8; do
    for seed in 1 2 3 4; do
        echo "[VAE-split default-credit-card missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/trainer.py \
            "data/default-credit-card/features-train.npy" \
            "data/default-credit-card/features-test.npy" \
            "data/default-credit-card/metadata.json" \
            "models/default-credit-card/vae-split/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/default-credit-card/vae-split/miss_prob${miss_prob}/seed${seed}/loss.csv" \
            "--max_seconds_without_save=300" \
            "--seed=${seed}" \
            "--data_format=dense" \
            "--start_epoch=0" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=93,46" \
            "--decoder_hidden_sizes=93,46" \
            "--batch_size=64" \
            "--num_epochs=1" \
            "--l2_regularization=0" \
            "--learning_rate=0.001" \
            "--missing_probability=${miss_prob}" \
            "--temperature=0.1"
    done

    for seed in 1 2 3 4; do
        echo "[VAE-split online-news-popularity missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/trainer.py \
            "data/online-news-popularity/features-train.npy" \
            "data/online-news-popularity/features-test.npy" \
            "data/online-news-popularity/metadata.json" \
            "models/online-news-popularity/vae-split/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/online-news-popularity/vae-split/miss_prob${miss_prob}/seed${seed}/loss.csv" \
            "--max_seconds_without_save=300" \
            "--seed=${seed}" \
            "--data_format=dense" \
            "--start_epoch=0" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=60,30" \
            "--decoder_hidden_sizes=60,30" \
            "--batch_size=64" \
            "--num_epochs=1" \
            "--l2_regularization=0" \
            "--learning_rate=0.001" \
            "--missing_probability=${miss_prob}" \
            "--temperature=0.1"
    done
done

########################################################################################################################
# VAE-iterative
########################################################################################################################

for miss_prob in 0.2 0.5 0.8; do
    for seed in 1 2 3 4; do
        echo "[VAE-iterative breast missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/iterative_imputation.py \
            "data/breast/features-test.npy" \
            "data/breast/metadata.json" \
            "models/breast/vae/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/breast/vae/miss_prob${miss_prob}/seed${seed}/iterative_loss.csv" \
            "--tolerance=0.001" \
            "--data_format=dense" \
            "--max_iterations=1" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=30,15" \
            "--decoder_hidden_sizes=30,15" \
            "--missing_probability=${miss_prob}"
    done

    for seed in 1 2 3 4; do
        echo "[VAE-iterative default-credit-card missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/iterative_imputation.py \
            "data/default-credit-card/features-test.npy" \
            "data/default-credit-card/metadata.json" \
            "models/default-credit-card/vae/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/default-credit-card/vae/miss_prob${miss_prob}/seed${seed}/iterative_loss.csv" \
            "--tolerance=0.001" \
            "--data_format=dense" \
            "--max_iterations=1" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=93,46" \
            "--decoder_hidden_sizes=93,46" \
            "--missing_probability=${miss_prob}"
    done

    for seed in 1 2 3 4; do
        echo "[VAE-iterative letter-recognition missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/iterative_imputation.py \
            "data/letter-recognition/features-test.npy" \
            "data/letter-recognition/metadata.json" \
            "models/letter-recognition/vae/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/letter-recognition/vae/miss_prob${miss_prob}/seed${seed}/iterative_loss.csv" \
            "--tolerance=0.001" \
            "--data_format=dense" \
            "--max_iterations=1" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=16,8" \
            "--decoder_hidden_sizes=16,8" \
            "--missing_probability=${miss_prob}"
    done

    for seed in 1 2 3 4; do
        echo "[VAE-iterative online-news-popularity missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/iterative_imputation.py \
            "data/online-news-popularity/features-test.npy" \
            "data/online-news-popularity/metadata.json" \
            "models/online-news-popularity/vae/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/online-news-popularity/vae/miss_prob${miss_prob}/seed${seed}/iterative_loss.csv" \
            "--tolerance=0.001" \
            "--data_format=dense" \
            "--max_iterations=1" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=60,30" \
            "--decoder_hidden_sizes=60,30" \
            "--missing_probability=${miss_prob}"
    done

    for seed in 1 2 3 4; do
        echo "[VAE-iterative spambase missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/iterative_imputation.py \
            "data/spambase/features-test.npy" \
            "data/spambase/metadata.json" \
            "models/spambase/vae/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/spambase/vae/miss_prob${miss_prob}/seed${seed}/iterative_loss.csv" \
            "--tolerance=0.001" \
            "--data_format=dense" \
            "--max_iterations=1" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=57,28" \
            "--decoder_hidden_sizes=57,28" \
            "--missing_probability=${miss_prob}"
    done
done

########################################################################################################################
# VAE-split-iterative
########################################################################################################################

for miss_prob in 0.2 0.5 0.8; do
    for seed in 1 2 3 4; do
        echo "[VAE-split-iterative default-credit-card missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/iterative_imputation.py \
            "data/default-credit-card/features-test.npy" \
            "data/default-credit-card/metadata.json" \
            "models/default-credit-card/vae-split/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/default-credit-card/vae-split/miss_prob${miss_prob}/seed${seed}/iterative_loss.csv" \
            "--tolerance=0.001" \
            "--data_format=dense" \
            "--max_iterations=1" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=93,46" \
            "--decoder_hidden_sizes=93,46" \
            "--missing_probability=${miss_prob}" \
            "--temperature=0.1"
    done

    for seed in 1 2 3 4; do
        echo "[VAE-split-iterative online-news-popularity missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/iterative_imputation.py \
            "data/online-news-popularity/features-test.npy" \
            "data/online-news-popularity/metadata.json" \
            "models/online-news-popularity/vae-split/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/online-news-popularity/vae-split/miss_prob${miss_prob}/seed${seed}/iterative_loss.csv" \
            "--tolerance=0.001" \
            "--data_format=dense" \
            "--max_iterations=1" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=60,30" \
            "--decoder_hidden_sizes=60,30" \
            "--missing_probability=${miss_prob}" \
            "--temperature=0.1"
    done
done

########################################################################################################################
# VAE-backprop
########################################################################################################################

for miss_prob in 0.2 0.5 0.8; do
    for seed in 1 2 3 4; do
        echo "[VAE-backprop breast missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/iterative_imputation.py \
            "data/breast/features-test.npy" \
            "data/breast/metadata.json" \
            "models/breast/vae/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/breast/vae/miss_prob${miss_prob}/seed${seed}/backprop_loss.csv" \
            "--noise_learning_rate=0.1" \
            "--tolerance=0.001" \
            "--data_format=dense" \
            "--max_iterations=1" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=30,15" \
            "--decoder_hidden_sizes=30,15" \
            "--missing_probability=${miss_prob}"
    done

    for seed in 1 2 3 4; do
        echo "[VAE-backprop default-credit-card missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/iterative_imputation.py \
            "data/default-credit-card/features-test.npy" \
            "data/default-credit-card/metadata.json" \
            "models/default-credit-card/vae/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/default-credit-card/vae/miss_prob${miss_prob}/seed${seed}/backprop_loss.csv" \
            "--noise_learning_rate=0.1" \
            "--tolerance=0.001" \
            "--data_format=dense" \
            "--max_iterations=1" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=93,46" \
            "--decoder_hidden_sizes=93,46" \
            "--missing_probability=${miss_prob}"
    done

    for seed in 1 2 3 4; do
        echo "[VAE-backprop letter-recognition missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/iterative_imputation.py \
            "data/letter-recognition/features-test.npy" \
            "data/letter-recognition/metadata.json" \
            "models/letter-recognition/vae/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/letter-recognition/vae/miss_prob${miss_prob}/seed${seed}/backprop_loss.csv" \
            "--noise_learning_rate=0.1" \
            "--tolerance=0.001" \
            "--data_format=dense" \
            "--max_iterations=1" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=16,8" \
            "--decoder_hidden_sizes=16,8" \
            "--missing_probability=${miss_prob}"
    done

    for seed in 1 2 3 4; do
        echo "[VAE-backprop online-news-popularity missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/iterative_imputation.py \
            "data/online-news-popularity/features-test.npy" \
            "data/online-news-popularity/metadata.json" \
            "models/online-news-popularity/vae/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/online-news-popularity/vae/miss_prob${miss_prob}/seed${seed}/backprop_loss.csv" \
            "--noise_learning_rate=0.1" \
            "--tolerance=0.001" \
            "--data_format=dense" \
            "--max_iterations=1" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=60,30" \
            "--decoder_hidden_sizes=60,30" \
            "--missing_probability=${miss_prob}"
    done

    for seed in 1 2 3 4; do
        echo "[VAE-backprop spambase missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/iterative_imputation.py \
            "data/spambase/features-test.npy" \
            "data/spambase/metadata.json" \
            "models/spambase/vae/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/spambase/vae/miss_prob${miss_prob}/seed${seed}/backprop_loss.csv" \
            "--noise_learning_rate=0.1" \
            "--tolerance=0.001" \
            "--data_format=dense" \
            "--max_iterations=1" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=57,28" \
            "--decoder_hidden_sizes=57,28" \
            "--missing_probability=${miss_prob}"
    done
done

########################################################################################################################
# VAE-split-backprop
########################################################################################################################

for miss_prob in 0.2 0.5 0.8; do
    for seed in 1 2 3 4; do
        echo "[VAE-split-backprop default-credit-card missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/iterative_imputation.py \
            "data/default-credit-card/features-test.npy" \
            "data/default-credit-card/metadata.json" \
            "models/default-credit-card/vae-split/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/default-credit-card/vae-split/miss_prob${miss_prob}/seed${seed}/backprop_loss.csv" \
            "--noise_learning_rate=0.1" \
            "--tolerance=0.001" \
            "--data_format=dense" \
            "--max_iterations=1" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=93,46" \
            "--decoder_hidden_sizes=93,46" \
            "--missing_probability=${miss_prob}" \
            "--temperature=0.1"
    done

    for seed in 1 2 3 4; do
        echo "[VAE-split-backprop online-news-popularity missing_probability=${miss_prob} seed=${seed}]"
        python imputation_dgm/methods/vae/iterative_imputation.py \
            "data/online-news-popularity/features-test.npy" \
            "data/online-news-popularity/metadata.json" \
            "models/online-news-popularity/vae-split/miss_prob${miss_prob}/seed${seed}/vae.torch" \
            "models/online-news-popularity/vae-split/miss_prob${miss_prob}/seed${seed}/backprop_loss.csv" \
            "--noise_learning_rate=0.1" \
            "--tolerance=0.001" \
            "--data_format=dense" \
            "--max_iterations=1" \
            "--split_size=30" \
            "--code_size=30" \
            "--encoder_hidden_sizes=60,30" \
            "--decoder_hidden_sizes=60,30" \
            "--missing_probability=${miss_prob}" \
            "--temperature=0.1"
    done
done