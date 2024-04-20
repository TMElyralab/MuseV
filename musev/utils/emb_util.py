import torch


def concat_two_text_embedding(
    emb1: torch.Tensor, emb2: torch.Tensor, dim: int, emb1_valid_length: int = None
) -> torch.Tensor:
    if emb1_valid_length is not None:
        emb1 = torch.index_select(
            emb1,
            dim=dim,
            index=torch.LongTensor(
                torch.arange(emb1_valid_length).to(device=emb1.device)
            ),
        )
    emb = torch.concat([emb1, emb2], dim=dim)
    return emb
