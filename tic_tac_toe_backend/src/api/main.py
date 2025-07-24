from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from sqlalchemy.exc import NoResultFound
from datetime import datetime

# Database setup
DATABASE_URL = "sqlite:///./tic_tac_toe.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Database Models
class Game(Base):
    __tablename__ = "games"
    id = Column(Integer, primary_key=True, index=True)
    player_x = Column(String, nullable=False)
    player_o = Column(String, nullable=False)
    board_state = Column(String, default=" " * 9)   # 9 space chars: empty board
    current_turn = Column(String, nullable=False, default="X")
    winner = Column(String, nullable=True)
    is_draw = Column(Integer, default=0)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    moves = relationship("Move", back_populates="game")


class Move(Base):
    __tablename__ = "moves"
    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"))
    player = Column(String, nullable=False)
    position = Column(Integer, nullable=False)
    move_time = Column(DateTime, default=datetime.utcnow)
    game = relationship("Game", back_populates="moves")


class Score(Base):
    __tablename__ = "scores"
    id = Column(Integer, primary_key=True, index=True)
    player = Column(String, nullable=False, index=True)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    draws = Column(Integer, default=0)


Base.metadata.create_all(bind=engine)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Pydantic Schemas
class GameCreate(BaseModel):
    player_x: str = Field(..., description="Name of Player X")
    player_o: str = Field(..., description="Name of Player O")


class MoveInput(BaseModel):
    game_id: int = Field(..., description="ID of the ongoing game")
    player: str = Field(..., description="Player making the move (X or O)")
    position: int = Field(..., description="Board position (0-8)")


class MoveResponse(BaseModel):
    board_state: str
    current_turn: str
    winner: Optional[str]
    is_draw: bool
    message: str


class GameHistoryItem(BaseModel):
    game_id: int
    player_x: str
    player_o: str
    winner: Optional[str]
    is_draw: bool
    start_time: datetime
    end_time: Optional[datetime]


class LeaderboardEntry(BaseModel):
    player: str
    wins: int
    losses: int
    draws: int


# FastAPI Setup
app = FastAPI(
    title="Tic Tac Toe Backend",
    description="Backend API for playing Tic Tac Toe, viewing game history, and leaderboard.",
    version="1.0.0",
    openapi_tags=[
        {"name": "Game", "description": "Tic Tac Toe game play routes."},
        {"name": "History", "description": "Game history routes."},
        {"name": "Leaderboard", "description": "Leaderboard and scores."},
    ],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set allowed frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Utility functions for game logic
def check_winner(board: str) -> Optional[str]:
    """Returns 'X', 'O', or None"""
    win_patterns = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
        [0, 4, 8], [2, 4, 6]              # diagonals
    ]
    for pattern in win_patterns:
        first = board[pattern[0]]
        if first != " " and all(board[i] == first for i in pattern):
            return first
    return None

def check_draw(board: str) -> bool:
    return ' ' not in board and check_winner(board) is None

def next_turn(board: str, current: str) -> str:
    return "O" if current == "X" else "X"


# API Endpoints

@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {"message": "Healthy"}


# PUBLIC_INTERFACE
@app.post("/game", response_model=GameHistoryItem, tags=["Game"], summary="Create new game", description="Start a new Tic Tac Toe game between two players.")
def create_game(game: GameCreate, db: Session = Depends(get_db)):
    """Create a new game given player names."""
    db_game = Game(player_x=game.player_x, player_o=game.player_o, current_turn="X")
    db.add(db_game)
    db.commit()
    db.refresh(db_game)
    return GameHistoryItem(
        game_id=db_game.id,
        player_x=db_game.player_x,
        player_o=db_game.player_o,
        winner=db_game.winner,
        is_draw=bool(db_game.is_draw),
        start_time=db_game.start_time,
        end_time=db_game.end_time
    )


# PUBLIC_INTERFACE
@app.post("/move", response_model=MoveResponse, tags=["Game"], summary="Make a move", description="Make a Tic Tac Toe move for an ongoing game.")
def make_move(move: MoveInput, db: Session = Depends(get_db)):
    """Make a move in a game, update game state, check for win/draw."""
    try:
        game = db.query(Game).filter(Game.id == move.game_id).one()
    except NoResultFound:
        raise HTTPException(status_code=404, detail="Game not found.")
    # Validate player
    if move.player != game.current_turn:
        raise HTTPException(status_code=400, detail="Not this player's turn.")
    # Validate position
    if move.position < 0 or move.position > 8:
        raise HTTPException(status_code=400, detail="Invalid board position.")
    if game.board_state[move.position] != " ":
        raise HTTPException(status_code=400, detail="Position already taken.")
    # Make move
    board = list(game.board_state)
    board[move.position] = move.player
    game.board_state = "".join(board)
    game.current_turn = next_turn(game.board_state, move.player)
    db.add(Move(game_id=move.game_id, player=move.player, position=move.position))
    # Check for winner
    winner = check_winner(game.board_state)
    draw = check_draw(game.board_state)
    msg = "Move accepted."
    if winner or draw:
        game.end_time = datetime.utcnow()
        if winner:
            game.winner = winner
            msg = f"Player {winner} wins!"
            inc_player = game.player_x if winner == "X" else game.player_o
            dec_player = game.player_x if winner == "O" else game.player_o
            # Update scores
            for username, is_win in [(inc_player, True), (dec_player, False)]:
                score = db.query(Score).filter(Score.player == username).one_or_none()
                if not score:
                    score = Score(player=username)
                    db.add(score)
                if is_win:
                    score.wins += 1
                else:
                    score.losses += 1
        elif draw:
            game.is_draw = 1
            msg = "Game is a draw."
            for player_name in [game.player_x, game.player_o]:
                score = db.query(Score).filter(Score.player == player_name).one_or_none()
                if not score:
                    score = Score(player=player_name)
                    db.add(score)
                score.draws += 1
    db.commit()
    return MoveResponse(
        board_state=game.board_state,
        current_turn=game.current_turn if not (winner or draw) else "",
        winner=winner,
        is_draw=draw,
        message=msg
    )


# PUBLIC_INTERFACE
@app.get("/history", response_model=List[GameHistoryItem], tags=["History"], summary="Get game history", description="Retrieve a list of previous games.")
def get_game_history(db: Session = Depends(get_db)):
    """Get list of completed and ongoing games."""
    games = db.query(Game).order_by(Game.start_time.desc()).limit(100).all()
    return [
        GameHistoryItem(
            game_id=g.id, player_x=g.player_x, player_o=g.player_o,
            winner=g.winner, is_draw=bool(g.is_draw),
            start_time=g.start_time, end_time=g.end_time
        )
        for g in games
    ]


# PUBLIC_INTERFACE
@app.get("/leaderboard", response_model=List[LeaderboardEntry], tags=["Leaderboard"], summary="Get leaderboard", description="Fetch top scores (wins, losses, draws) sorted by wins.")
def get_leaderboard(db: Session = Depends(get_db)):
    """Fetch top players sorted by wins."""
    scores = db.query(Score).order_by(Score.wins.desc(), Score.draws.desc()).limit(20).all()
    return [LeaderboardEntry(player=s.player, wins=s.wins, losses=s.losses, draws=s.draws) for s in scores]


# PUBLIC_INTERFACE
@app.get("/game/{game_id}", tags=["Game"], summary="Get the current state of a game", description="Check board and status for an existing game.")
def get_game_state(game_id: int, db: Session = Depends(get_db)):
    """Get board state and game status for a specific game."""
    try:
        game = db.query(Game).filter(Game.id == game_id).one()
    except NoResultFound:
        raise HTTPException(status_code=404, detail="Game not found.")
    history = [
        {"player": m.player, "position": m.position, "move_time": m.move_time}
        for m in db.query(Move).filter(Move.game_id == game_id).order_by(Move.id)
    ]
    return {
        "game_id": game.id,
        "player_x": game.player_x,
        "player_o": game.player_o,
        "board_state": game.board_state,
        "current_turn": game.current_turn,
        "winner": game.winner,
        "is_draw": bool(game.is_draw),
        "start_time": game.start_time,
        "end_time": game.end_time,
        "history": history,
    }

