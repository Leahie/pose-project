import redis 
import pickle

r = redis.Redis(host="localhost", port=6379)

def save_dataset(dataset_id, dataset):
    r.set(dataset_id, pickle.dumps(dataset), ex=1800)

def load_dataset(dataset_id):
    data = r.get(dataset_id)
    return pickle.loads(data)