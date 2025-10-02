"""
Dataset for Continual Pre-Training (CPT)
Unlike SFT, CPT trains on ALL tokens (no masking)
"""
import torch
from torch.utils.data import Dataset
import ujson as json
from typing import Dict


class CPTDataset(Dataset):
    """Dataset for Continual Pre-Training (text-only)"""
    
    def __init__(
        self,
        data_path: str,
        processor,
        data_args,
        model_id: str,
    ):
        super().__init__()
        
        if isinstance(data_path, str):
            self.list_data_dict = json.load(open(data_path, "r"))
        else:
            self.list_data_dict = data_path
        
        self.processor = processor
        self.model_id = model_id
        self.max_length = getattr(data_args, 'max_seq_length', 4096)
        
        print(f"CPTDataset initialized:")
        print(f"  Samples: {len(self.list_data_dict)}")
        print(f"  Max length: {self.max_length}")
    
    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[i]
        text = sample["text"]
        
        # Tokenize the text
        encoded = self.processor.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        
        # KEY DIFFERENCE FROM SFT: Labels = input_ids (train on ALL tokens!)
        labels = input_ids.clone()
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class CPTDataCollator:
    """Collator for CPT dataset with padding"""
    
    def __init__(self, pad_token_id: int, padding_side: str = "right"):
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
    
    def __call__(self, examples):
        # Find max length in batch
        max_len = max(ex["input_ids"].size(0) for ex in examples)
        
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        
        for ex in examples:
            seq_len = ex["input_ids"].size(0)
            padding_len = max_len - seq_len
            
            if self.padding_side == "right":
                # Pad on the right
                padded_input = torch.cat([
                    ex["input_ids"],
                    torch.full((padding_len,), self.pad_token_id, dtype=torch.long)
                ])
                
                # Pad labels with -100 so they're ignored in loss
                padded_labels = torch.cat([
                    ex["labels"],
                    torch.full((padding_len,), -100, dtype=torch.long)
                ])
                
                padded_mask = torch.cat([
                    ex["attention_mask"],
                    torch.zeros(padding_len, dtype=torch.long)
                ])
            else:
                # Pad on the left
                padded_input = torch.cat([
                    torch.full((padding_len,), self.pad_token_id, dtype=torch.long),
                    ex["input_ids"]
                ])
                
                padded_labels = torch.cat([
                    torch.full((padding_len,), -100, dtype=torch.long),
                    ex["labels"]
                ])
                
                padded_mask = torch.cat([
                    torch.zeros(padding_len, dtype=torch.long),
                    ex["attention_mask"]
                ])
            
            batch_input_ids.append(padded_input)
            batch_labels.append(padded_labels)
            batch_attention_mask.append(padded_mask)
        
        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_attention_mask),
        }


def make_cpt_data_module(model_id, processor, data_args):
    """Make dataset and collator for CPT"""
    
    # Create training dataset
    train_dataset = CPTDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        model_id=model_id
    )
    
    # Create evaluation dataset if provided
    eval_dataset = None
    if hasattr(data_args, 'eval_path') and data_args.eval_path:
        eval_dataset = CPTDataset(
            data_path=data_args.eval_path,
            processor=processor,
            data_args=data_args,
            model_id=model_id
        )
        print(f"  Eval samples: {len(eval_dataset)}")
    
    # Create data collator
    data_collator = CPTDataCollator(
        pad_token_id=processor.tokenizer.pad_token_id,
        padding_side="right"
    )
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
