Network BERT_BASE {
    Layer BERT_FFN_1 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 3072, C: 768, R: 1, S: 1, Y: 1, X: 1 }
Dataflow {
    TemporalMap(16,16) K;
    SpatialMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(1,1) C;
    Cluster(16, P);
    SpatialMap(1,1) K;
    TemporalMap(Sz(R),1) Y;
    TemporalMap(Sz(S),1) X;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
    }
}