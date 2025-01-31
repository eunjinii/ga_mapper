Network BERT_BASE {
    Layer BERT_QK {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 64, R: 1, S: 1, Y: 128, X: 1 }
    }
}