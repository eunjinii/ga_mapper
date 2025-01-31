Network BERT_BASE {
    Layer BERT_FFN_1 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 3072, C: 768, R: 1, S: 1, Y: 1, X: 1 }
    }
}