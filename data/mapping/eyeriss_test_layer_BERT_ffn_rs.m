Network BERT_BASE {
    Layer BERT_FFN_1 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 3072, C: 768, R: 1, S: 1, Y: 1, X: 1 }
 Dataflow {
     // This is an Eyeriss-like row-stationary dataflow” 
    SpatialMap(1,1) Y';
    TemporalMap(1,1) X';
    TemporalMap(1,1) C;
    TemporalMap(16,16) K;
    TemporalMap(Sz(R),Sz(R)) R;
    TemporalMap(Sz(S),Sz(S)) S;
    Cluster(Sz(R),P);
    SpatialMap(1,1) Y;
    SpatialMap(1,1) R;
    TemporalMap(Sz(S),Sz(S)) S;
}
    }
}