from cebra_lens.model import model_loader
import pytest

#learn how to write unittests, the work flow and using assert, and @tags

def test_model_loader():
    models = model_loader("D:\EPFL\MA2\project\FinalModels\VISION\offset10")


def main():
    test_model_loader()