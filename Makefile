.PHONY: build run1 run2 run3 inference

# Exécution en arrière-plan (-d) avec bash -c pour enchaîner les scripts
TRAINER_BG := docker compose run -d --rm trainer bash -c
# Le clean reste bloquant (sans -d) pour s'assurer que le nettoyage est fini avant l'entraînement
CLEAN      := docker compose run --rm --no-deps trainer bash -c

build:
	docker compose down
	docker compose build

run1:
	docker compose stop trainer
	$(CLEAN) "rm -rf /app/data/* /app/checkpoints/* /app/runs/*"
	$(TRAINER_BG) "python src/1_collect_data.py && python src/2_train_world.py && python src/3_train_controller.py"

run2:
	docker compose stop trainer
	$(CLEAN) "rm -rf /app/checkpoints/* /app/runs/*"
	$(TRAINER_BG) "python src/2_train_world.py && python src/3_train_controller.py"

run3:
	docker compose stop trainer
	$(CLEAN) "rm -f /app/checkpoints/controller.pt && rm -rf /app/runs/controller /app/runs/status.json"
	$(TRAINER_BG) "python src/3_train_controller.py"

inference:
	docker compose stop inference
	docker compose up -d inference
	