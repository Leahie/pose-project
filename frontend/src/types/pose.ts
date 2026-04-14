export interface PoseType{
    id: string;
    points: number[][]; // [[x,y,z][x,y,z]]
    visibility: number[]; // visibility for each feature
    metadata: {
        png?: string;
        createdAt?: Date;
        title?: string;
    }
}