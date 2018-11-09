class AuthorConfig:
    batch_size=64
    num_steps=40
    num_epochs=4000
    rnn_size=100
    num_layers=3
    max_grad_norm=5
    learning_rate = 0.001
    embed_size=100
    file_path = 'jane\jane_all.txt'
    outfile_name = 'jane_all2'
    temp = 0.95
