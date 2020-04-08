import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def get_last_layer_attention(model,tokenizer,sample):
    seq,attn_mask,label = sample
    seq = seq.view(1,-1).cuda()
    attn_mask = attn_mask.view(1,-1).cuda()
    #forward pass
    output = model(seq,attn_mask)
    #retrieve attentions
    attention = output[-1]
    input_id_list = seq[0].tolist() # Batch index 0
    #convert to human-readable
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    attn_data = {'layer'+str(i):a[0] for i,a in enumerate(attention) }
    mean_attn = attn_data['layer11'].cpu().detach().numpy().mean(axis=0).mean(axis=0)

    return mean_attn,tokens,attn_data


def visualize_layer_attention(layer_attn):
    fig = plt.figure(figsize=(14,5))
    for i in range(12):
        ax = fig.add_subplot(2,6,i+1)
        ax.imshow(layer_attn[i])

def get_words_level_attentions(tokens,array,tokenizer):
    mask = np.array([t not in ['<s>','<pad>','</s>'] for t in tokens])
    tokens = np.asarray(tokens)[mask]
    array = array[mask]

    attentions = []
    index = 0
    SPIECE_UNDERLINE = "▁"
    for i,token in enumerate(tokens):
        if token.startswith(SPIECE_UNDERLINE):
            attentions.append(0)
        attentions[-1] += array[i]

    words = tokenizer.convert_tokens_to_string(tokens).split()
    assert len(attentions) == len(words)
    return np.array(attentions), words

def display_attention(tokens,array,tokenizer):

    attention, words = get_words_level_attentions(tokens,array,tokenizer)
    #define cmap and normalize array
    cmap = matplotlib.cm.get_cmap('Reds')
    norm_array = (attention-attention.min())/(attention.max()-attention.min())
    # Pixels where you want to start first word
    start_x = 20
    start_y = 100
    # Decide what is the last pixel for writing words in one line
    end = 600
    # Whitespace in pixels
    whitespace = 8

    # Create figure
    figure = plt.figure(figsize=(14,2))
    # From renderer we can get textbox width in pixels
    rend = figure.canvas.get_renderer()
    for i,word in enumerate(words):
        if word !='<s>':
            # Text box parameters and colors
            col = cmap(norm_array[i])
            bbox = dict(boxstyle="round,pad=0.3", fc=col, ec='w',lw=2)
            txt = plt.text(start_x, start_y, word, color="black", bbox=bbox,transform=None)
            # Textbox width
            bb = txt.get_window_extent(renderer=rend)

            # Calculate where next word should be written
            start_x = bb.width + start_x + whitespace

            # Next line if end parameter in pixels have been crossed
            if start_x >= end:
                start_x = 20
                start_y -= 40

    # Skip plotting axis
    plt.axis("off")
    plt.show()
