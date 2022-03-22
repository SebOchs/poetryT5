import gpt_2_simple as gpt2
import os


def finetune():
    model_name = "124M"
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(
            model_name=model_name
        )  # Model is saved into current directory under /models/124M/

    # Dataset for training
    file_name = "data.txt"

    sess = gpt2.start_tf_sess()

    gpt2.finetune(
        sess,
        file_name,
        model_name=model_name,
        checkpoint_dir="checkpoint",
        batch_size=2,
        accumulate_gradients=32,
        learning_rate=0.0001,
        sample_every=50,
        sample_length=200,
        save_every=100,
        steps=1000,
    )  # Steps is max number of training steps


def generate():
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess)

    gpt2.generate(
        sess,
        prefix="abba",
        length=50,
        sample_delim="\n",
        nsamples=3000,
        destination_path="samples/abba.txt",
    )
