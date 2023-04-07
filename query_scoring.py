import torch


def similarity_queryscoring(text_embeds, vid_embeds, temperature=0.1, eps=1e-8, mean='feats'):
    """
    :param text_embeds: Tensor of shape [b_t, d]
    :param vid_embeds: Tensor of shape [b_v, v, d]
    :param temperature: float
    :param eps: float
    :param mean: str 'feats' or 'scores', what to average over. Query-scoring default SOTA is mean over feats
    :return: Tensor, Tensor of shape [b_t, b_v] and [b_v, b_t]. Similarity scores.
    """
    assert len(text_embeds.shape) == 2
    assert len(vid_embeds.shape) == 3

    t_b_n, v_b_n = text_embeds.norm(dim=1)[:, None], vid_embeds.norm(dim=2)[:, :, None]
    text_embeds_norm = text_embeds / torch.max(t_b_n, eps * torch.ones_like(t_b_n))
    vid_embeds_norm = vid_embeds / torch.max(v_b_n, eps * torch.ones_like(v_b_n))

    # calculate similarity between text embeds and every frame embedding
    sim_mt = torch.einsum('a d, b v d -> a b v', text_embeds_norm, vid_embeds_norm)

    # compute "query-scores", temperature is variable parameter
    scores = torch.softmax(sim_mt / temperature, dim=2)

    if mean == 'feats':
        # weighted sum of video embeds * scores, producing single video embedding
        vid_embeds_weighted = torch.einsum('b v d, a b v -> a b v d', vid_embeds, scores)
        vid_embed_final = vid_embeds_weighted.sum(dim=2)

        # normalize
        v_f_n = vid_embed_final.norm(dim=2)[:, :, None]
        vid_embed_final_norm = vid_embed_final / torch.max(v_f_n, eps * torch.ones_like(v_f_n))

        # calculate similarity score between text
        weighted_sim_mt = torch.einsum('a d, a b d -> a b', text_embeds_norm, vid_embed_final_norm)
    elif mean == 'scores':
        v_b_n = vid_embeds.norm(dim=2, keepdim=True)
        vid_embeds_norm = vid_embeds / torch.max(v_b_n, eps * torch.ones_like(v_b_n))
        weighted_sim_mt = torch.einsum('a d, b v d -> a b v', text_embeds_norm, v_b_n)
        weighted_sim_mt = weighted_sim_mt * scores
        weighted_sim_mt = weighted_sim_mt.sum(dim=2)
    else:
        raise ValueError

    return weighted_sim_mt

def similarity_meanpooling(text_embeds, vid_embeds, eps=1e-8):
    """
    :param text_embeds: Tensor of shape [b_t, d]
    :param vid_embeds: Tensor of shape [b_v, v, d]
    :param eps: float
    :return: Tensor, Tensor of shape [b_t, b_v] and [b_v, b_t]. Similarity scores.
    """
    vid_embeds = vid_embeds.mean(1)

    t_b_n, v_b_n = text_embeds.norm(dim=1)[:, None], vid_embeds.norm(dim=1)[:, None]
    text_embeds_norm = text_embeds / torch.max(t_b_n, eps * torch.ones_like(t_b_n))
    vid_embeds_norm = vid_embeds / torch.max(v_b_n, eps * torch.ones_like(v_b_n))

    sim_mt = torch.einsum('a d, b d -> a b', text_embeds_norm, vid_embeds_norm)
    return sim_mt