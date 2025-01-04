import skein as pyskein
import os
from hashlib import sha256
from Crypto.Protocol.KDF import scrypt
from math import tan, factorial, isclose
import numpy as np
import time

# copilot adding comments go brrrrr
# Advanced Logistic Map
def logistic_map(x, r=3.99):
    return r * x * (1 - x)

# Advanced Chaos Function (tan(x * y!))
def chaos_function(x, y):
    try:
        result = tan(x * factorial(y))
        return result if not isclose(result, 1, abs_tol=1e-9) else 0.999
    except OverflowError:
        return 0.999

# random mask
def secure_random_mask(length=128):
    return os.urandom(length)

# entropy conditions
def domain_separation(state, domain="default"):
    domain_bytes = domain.encode('utf-8')
    state = state + list(domain_bytes)
    return state

# memory-hard KDF
def memory_hard_kdf(data, salt, N=2**14, r=8, p=1, dklen=64):
    return scrypt(data, salt, dklen, N, r, p)

# simple lattice-based KDF
def lattice_based_kdf(data, salt, modulus=257, dimension=512):
    # make sure matrix stuff line up
    data_ints = [byte % modulus for byte in data]
    salt_ints = [byte % modulus for byte in salt]

    # pad salt_ints to match the dimension of the lattice if its needed
    if len(salt_ints) < dimension:
        salt_ints = salt_ints + [0] * (dimension - len(salt_ints))

    # Create basis matrix (dimension x len(data_ints)) and noise vector
    basis_matrix = np.random.randint(0, modulus, (dimension, len(data_ints)))
    noise_vector = np.random.randint(-10, 10, dimension)

    # make sure stuff are the same length
    lattice_vector = (np.dot(basis_matrix, data_ints) + salt_ints[:dimension] + noise_vector) % modulus
    
    # hash it with sha256
    lattice_hash = sha256(lattice_vector.tobytes()).digest()
    return lattice_hash


# stupid chaos func
def apply_nonlinear_state_update(state, r=3.99):
    state_floats = [logistic_map(byte / 255.0, r) for byte in state]
    state_chaos = [
        chaos_function(state_floats[i % len(state_floats)], i + 1)
        for i in range(len(state))
    ]
    state_ints = [int(min(255, max(0, round(value * 255)))) for value in state_chaos]
    return state_ints

# math stuff (heegaard splitting)
def heegaard_splitting(state):
    midpoint = len(state) // 2
    handlebody1 = state[:midpoint]
    handlebody2 = state[midpoint:]
    handlebody1 = apply_nonlinear_state_update(handlebody1)
    handlebody2 = apply_nonlinear_state_update(handlebody2)
    recombined = []
    for i in range(max(len(handlebody1), len(handlebody2))):
        if i < len(handlebody1):
            recombined.append(handlebody1[i])
        if i < len(handlebody2):
            recombined.append(handlebody2[i])

    return recombined
# weird tqft invariants, helps for quantum resistance i think
def tqft_invariants(state):
    # braid transform but simpler
    braid_transform = lambda x, y: (x * y + 7) % 257 # 257 is that one number that looks completely random but actually makes sense :speaking_head:
    transformed_state = [
        braid_transform(state[i], state[i - 1] if i > 0 else state[-1])
        for i in range(len(state))
    ]
    return transformed_state

# recursive hashing to mess with my cooling fans
def recursive_hash(state, rounds=5, r=3.99, lattice_kdf=None, memory_hard_func=None):
    for i in range(rounds):
        state = heegaard_splitting(state)  # Apply non-standard Heegaard splitting
        state = tqft_invariants(state)
        state = apply_nonlinear_state_update(state, r)
        if lattice_kdf:
            state = lattice_kdf(state)
        if memory_hard_func:
            state = memory_hard_func(state)
    return state
def apply_mask(state, mask):
    return [x ^ mask[i % len(mask)] for i, x in enumerate(state)]

def custom_hash(input_data, salt, rounds=5, domain="default", twf_factor=10000, memory_hard_func=None, lattice_kdf=None):
    # Adjust rounds using tunable work factor
    adjusted_rounds = twf_factor * rounds

    # Initialize state
    state = list(input_data.encode('utf-8'))
    state += list(salt)
    state = domain_separation(state, domain)

    # Recursive hashing with TQFT, Heegaard splitting, lattice KDF
    state = recursive_hash(state, adjusted_rounds, lattice_kdf=lattice_kdf, memory_hard_func=memory_hard_func)

    # Apply secure masking
    mask = secure_random_mask(len(state))
    state = apply_mask(state, mask)

    # Final hashing using Skein-1024
    state = state[:128]
    state = state + [0] * (128 - len(state))  # Pad if necessary
    hashed_output = pyskein.skein1024(bytes(state)).hexdigest()

    return hashed_output

memory_hard_func = lambda x: memory_hard_kdf(bytes(x), salt)
lattice_kdf = lambda x: lattice_based_kdf(x, salt)
# Example usage
if __name__ == "__main__":
    start = time.time()
    input_data = "idk"
    salt = os.urandom(64)

    hashed_output = custom_hash(
        input_data,
        salt,
        memory_hard_func=memory_hard_func,
        lattice_kdf=lattice_kdf
    )
    end = time.time()
    time = end - start
    print(f"Hashed Output: {hashed_output} | Time: {time} seconds")
