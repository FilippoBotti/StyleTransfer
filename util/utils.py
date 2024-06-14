import torch
import models.StyTR as StyTR
import torch.nn as nn
import models.transformer as transformer

def load_pretrained(args, sota=False):
    vgg = StyTR.vgg
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    if sota:
        decoder = StyTR.decoder_sota
        Trans = transformer.Transformer(args=None)
        decoder_path = args.decoder_sota_path
        trans_path = args.Trans_sota_path
        embedding_path = args.embedding_sota_path
    else:
        decoder = StyTR.decoder
        Trans = transformer.Transformer(args=args)
        decoder_path = args.decoder_path
        trans_path = args.Trans_path
        embedding_path = args.embedding_path
        
    embedding = StyTR.PatchEmbed()

    decoder.eval()
    Trans.eval()
    vgg.eval()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load(decoder_path)
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    decoder.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(trans_path)
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    Trans.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(embedding_path)
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    embedding.load_state_dict(new_state_dict)

    if sota:
        with torch.no_grad():
            network = StyTR.StyTrans(vgg,decoder,embedding,Trans,None)
    else:
        with torch.no_grad():
            network = StyTR.StyTrans(vgg,decoder,embedding,Trans,args)
    
    print("Loaded checkpoints!")
    return network