"""Shared local/API embedding adapters for indexing and retrieval."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any
from src.common.generation import load_env

DEFAULT_MODEL = 'Qwen/Qwen3-Embedding-4B'
DEFAULT_QUERY_INSTRUCTION = 'Given an analytical search question, retrieve passages that contain the evidence needed to answer it.'


def add_arguments(parser):
    load_env()
    parser.add_argument('--embedding-backend', choices=('local','api'), default=os.getenv('EMBEDDING_BACKEND','local'))
    parser.add_argument('--embedding-model', default=os.getenv('EMBEDDING_MODEL') or DEFAULT_MODEL)
    parser.add_argument('--embedding-dim', type=int, default=int(os.getenv('EMBEDDING_DIM','1024')))
    parser.add_argument('--embedding-base-url', default=os.getenv('EMBEDDING_BASE_URL'), help='API base URL, including /v1 if required')
    parser.add_argument('--embedding-api-key', default=None, help='Override EMBEDDING_API_KEY; omit for unauthenticated local servers')
    parser.add_argument('--embedding-timeout', type=float, default=120)
    parser.add_argument('--embedding-max-retries', type=int, default=3, help='Additional retries for transient API errors')
    parser.add_argument('--embedding-send-dimensions', action='store_true', help='Send dimensions to APIs that support this optional field')
    parser.add_argument('--embedding-document-prefix', default='', help='Prefix for document embeddings, e.g. passage: ')


def normalized(vectors, count, dimension):
    import numpy as np
    matrix=np.asarray(vectors,dtype='float32')
    if matrix.shape != (count,dimension):
        raise ValueError(f'Embedding shape {matrix.shape}; expected {(count,dimension)}. Set --embedding-dim to the model output dimension.')
    if not np.isfinite(matrix).all():raise ValueError('Embedding contains non-finite values')
    norms=np.linalg.norm(matrix,axis=1,keepdims=True)
    if np.any(norms==0):raise ValueError('Embedding returned a zero vector')
    return np.ascontiguousarray(matrix/np.maximum(norms,1e-12))


class LocalEmbeddingClient:
    def __init__(self,model,dimension,max_length,device,local_files_only=False,attn_implementation='auto'):
        self.model=model;self.dimension=dimension;self.max_length=max_length
        self.device=device;self.local_files_only=local_files_only;self.attn_implementation=attn_implementation
        self._encoder=None

    def _load(self):
        if self._encoder is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:raise RuntimeError('Install torch and sentence-transformers for local embeddings.') from exc
            if self.device.startswith('cuda') and not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable; use --embedding-device cpu (or --device cpu when building).')
            kwargs={} if self.attn_implementation=='auto' else {'attn_implementation':self.attn_implementation}
            self._encoder=SentenceTransformer(self.model,device=self.device,model_kwargs=kwargs,
                truncate_dim=self.dimension,local_files_only=self.local_files_only)
            self._encoder.max_seq_length=self.max_length
        return self._encoder

    def embed(self,texts):
        if not texts:
            import numpy as np
            return np.empty((0,self.dimension),dtype='float32')
        vectors=self._load().encode(texts,batch_size=min(32,len(texts)),show_progress_bar=False,
            convert_to_numpy=True,normalize_embeddings=True,prompt='')
        return normalized(vectors,len(texts),self.dimension)


class APIEmbeddingClient:
    def __init__(self,model,dimension,base_url,api_key='',timeout=120,max_retries=3,send_dimensions=False):
        if not base_url:raise ValueError('Set EMBEDDING_BASE_URL or --embedding-base-url for the API backend.')
        self.model=model;self.dimension=dimension;self.base_url=base_url.rstrip('/')
        self.api_key=api_key;self.timeout=timeout;self.max_retries=max_retries;self.send_dimensions=send_dimensions

    def embed(self,texts):
        if not texts:
            import numpy as np
            return np.empty((0,self.dimension),dtype='float32')
        payload={'model':self.model,'input':texts,'encoding_format':'float'}
        if self.send_dimensions:payload['dimensions']=self.dimension
        headers={'Content-Type':'application/json'}
        if self.api_key:headers['Authorization']='Bearer '+self.api_key
        for attempt in range(self.max_retries+1):
            request=urllib.request.Request(self.base_url+'/embeddings',data=json.dumps(payload).encode(),headers=headers,method='POST')
            try:
                with urllib.request.urlopen(request,timeout=self.timeout) as response:result=json.load(response)
                data=result.get('data',[])
                if len(data)!=len(texts) or sorted(row['index'] for row in data)!=list(range(len(texts))):
                    raise ValueError('Embedding API returned missing, duplicate, or invalid input indices')
                vectors=[row['embedding'] for row in sorted(data,key=lambda row:row['index'])]
                return normalized(vectors,len(texts),self.dimension)
            except urllib.error.HTTPError as exc:
                if exc.code not in (408,429,500,502,503,504) or attempt==self.max_retries:
                    raise RuntimeError(f'Embedding API HTTP {exc.code}') from None
            except (urllib.error.URLError,TimeoutError):
                if attempt==self.max_retries:raise RuntimeError('Embedding API connection failed or timed out') from None
            time.sleep(min(2**attempt,8))


def create_client(args,build=False):
    if args.embedding_dim<1:raise ValueError('--embedding-dim must be positive')
    if args.embedding_backend=='api':
        if args.embedding_timeout<=0 or args.embedding_max_retries<0:raise ValueError('Invalid embedding API timeout/retry count')
        return APIEmbeddingClient(args.embedding_model,args.embedding_dim,args.embedding_base_url,
            args.embedding_api_key or os.getenv('EMBEDDING_API_KEY',''),args.embedding_timeout,
            args.embedding_max_retries,args.embedding_send_dimensions)
    return LocalEmbeddingClient(args.embedding_model,args.embedding_dim,
        args.max_length if build else args.embedding_max_length,args.device if build else args.embedding_device,
        args.local_files_only if build else args.embedding_local_files_only,
        args.attn_implementation if build else args.embedding_attn_implementation)


def signature(args):
    result={'embedding_backend':args.embedding_backend,'document_prefix':args.embedding_document_prefix}
    if args.embedding_backend=='local':result['document_max_length']=getattr(args,'max_length',getattr(args,'embedding_max_length',1024))
    if args.embedding_backend=='api':result['embedding_base_url']=(args.embedding_base_url or '').rstrip('/')
    return result


def query_text(args,query):
    instruction=args.query_instruction
    if instruction is None:instruction=DEFAULT_QUERY_INSTRUCTION if 'qwen3-embedding' in args.embedding_model.lower() else ''
    if instruction:return f'Instruct: {instruction}\nQuery: {query}'
    return (getattr(args,'embedding_query_prefix','') or '')+query
