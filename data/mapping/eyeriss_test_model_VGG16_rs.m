Network YOLOv8 {
    Layer Conv_0 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 3, R: 3, S: 3, Y: 224, X: 224 }
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
    Layer Conv_2 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 224, X: 224 }
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
    Layer Conv_5 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 64, R: 3, S: 3, Y: 112, X: 112 }
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
    Layer Conv_7 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 128, C: 128, R: 3, S: 3, Y: 112, X: 112 }
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
    Layer Conv_10 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 256, C: 128, R: 3, S: 3, Y: 56, X: 56 }
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
    Layer Conv_fc1 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 4096, C: 512, R: 7, S: 7, Y: 7, X: 7 }
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
    Layer Conv_fc2 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1, X: 1 }
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
    Layer Linear {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 1000, C: 4096, R: 1, S: 1, Y: 1, X: 1 }
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