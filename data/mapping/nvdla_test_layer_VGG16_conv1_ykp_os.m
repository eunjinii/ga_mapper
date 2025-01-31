Network VGG16 {
  Layer VGG16_layer1 {
    Type: CONV
    Stride { X: 1, Y: 1 }
    Dimensions { K: 64, C: 3, R: 3, S: 3, Y: 224, X: 224 }
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