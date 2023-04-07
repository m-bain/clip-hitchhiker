import argparse
from tqdm import tqdm
import clip
import torch
from torch.utils.data import DataLoader

from query_scoring import similarity_queryscoring, similarity_meanpooling
from dataset import MSRVTT_dataset
from metrics import t2v_metrics


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    dataset = MSRVTT_dataset(preprocess, args.num_frames, "test")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    video_features_arr = []
    text_features_arr = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            text = clip.tokenize(batch['text']).to(device)

            text_features = model.encode_text(text)

            video = batch['video'].to(device)
            batch_sz, num_frames, channels, height, width = video.shape
            video = video.view(batch_sz * num_frames, channels, height, width)

            video_features = model.encode_image(video)

            video_features = video_features.view(batch_sz, num_frames, video_features.shape[1])

            text_features_arr.append(text_features.cpu())
            video_features_arr.append(video_features.cpu())

    text_features_arr = torch.cat(text_features_arr, dim=0).float()
    video_features_arr = torch.cat(video_features_arr, dim=0).float()

    sim_qs = similarity_queryscoring(text_features_arr, video_features_arr, temperature=args.query_scoring_temp)
    sim_mean = similarity_meanpooling(text_features_arr, video_features_arr)

    metrics_qs = t2v_metrics(sim_qs)
    metrics_mean = t2v_metrics(sim_mean)

    print(metrics_mean, " agg: mean-pooling")
    print(metrics_qs, f" agg: query-scoring, temp: {args.query_scoring_temp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot text-to-video retrieval performance of CLIP on MSR-VTT 1k-A")
    parser.add_argument('--num_frames', default=120, type=int)
    parser.add_argument('--query_scoring_temp', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=1, type=int)

    args = parser.parse_args()
    main(args)
