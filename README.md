# Research
GNN
<img width="1967" height="495" alt="image" src="https://github.com/user-attachments/assets/d844adaa-2577-45df-a47c-227f1536a89b" />

Heatmap Model
<img width="1561" height="985" alt="image" src="https://github.com/user-attachments/assets/e2118a83-b129-4313-a1f7-cca2c5d74121" />

# Frontend
For future reference: 
- Each URL page should be a new folder 
- Inside each folder the shared page is `folder_name`.tsx
- Shared components are within the folder `components`
- Major components with multiple components within them are given folders 

# Backend
For future reference:
- Each API resource should have its own route file in `app/api/routes` (e.g., datasets, poses, projects)
- Keep `main.py` minimal: app setup, middleware, and router registration only
- Business logic belongs in `app/services`; route handlers should stay thin
- External systems and data access belong in `app/repositories` (database, Redis, S3)
- Request and response contracts belong in `app/schemas`; persistence entities belong in `app/models`
