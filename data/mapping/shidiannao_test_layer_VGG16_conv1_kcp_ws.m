Network VGG16 {
  Layer VGG16_layer1 {
    Type: CONV
    Stride { X: 1, Y: 1 }
    Dimensions { K: 64, C: 3, R: 3, S: 3, Y: 224, X: 224 }
Dataflow {
        // This is a NVDLA-like dataflow
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
}
  }
}