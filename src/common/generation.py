"""One generation endpoint shared by every baseline."""
import os
from pathlib import Path


def load_env():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=False)


def resolve(args=None, required=True):
    load_env()
    def value(flag, env):
        return (getattr(args, flag, None) if args is not None else None) or os.getenv(env, '')
    model=value('model','GENERATION_MODEL')
    base=value('base_url','GENERATION_BASE_URL')
    key=value('api_key','GENERATION_API_KEY')
    if required and not all((model,base,key)):
        raise ValueError('Set GENERATION_BASE_URL, GENERATION_API_KEY, and GENERATION_MODEL in .env.')
    return model,base,key


def protocol_for(model):
    name=model.lower()
    return 'qwen' if 'qwen' in name else 'deepseek' if 'deepseek' in name else 'compatible'


def model_config(args=None, required=True):
    model,base,key=resolve(args,required)
    protocol=protocol_for(model)
    parameters={'temperature':0.0}
    if protocol=='qwen':
        parameters={'enable_thinking':True,'temperature':1.0,'top_p':0.95,'top_k':20,'presence_penalty':1.5,'repetition_penalty':1.0}
    elif protocol=='deepseek':
        parameters={'thinking':{'type':'enabled'},'reasoning_effort':'high'}
    return {'model':model,'base_url':base,'key_env':'GENERATION_API_KEY','protocol':protocol,'parameters':parameters},key
