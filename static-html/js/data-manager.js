// Data Manager for Horse Racing Prediction App
class DataManager {
    // Race management
    static addRace(raceData) {
        try {
            const races = this.getRaces();
            raceData.id = raceData.id || Date.now().toString();
            raceData.createdAt = new Date().toISOString();
            races.push(raceData);
            localStorage.setItem('races', JSON.stringify(races));
            return true;
        } catch (error) {
            console.error('Error adding race:', error);
            return false;
        }
    }

    static getRaces() {
        try {
            return JSON.parse(localStorage.getItem('races') || '[]');
        } catch (error) {
            console.error('Error loading races:', error);
            return [];
        }
    }

    static getRaceById(id) {
        const races = this.getRaces();
        return races.find(race => race.id === id);
    }

    static getUpcomingRaces() {
        const races = this.getRaces();
        const now = new Date();
        
        // Filter races that are in the future
        return races.filter(race => {
            const raceDate = new Date(race.date);
            return raceDate > now;
        }).sort((a, b) => new Date(a.date) - new Date(b.date)); // Sort by date ascending
    }

    static updateRace(id, updates) {
        try {
            const races = this.getRaces();
            const index = races.findIndex(race => race.id === id);
            if (index !== -1) {
                races[index] = { ...races[index], ...updates };
                localStorage.setItem('races', JSON.stringify(races));
                return true;
            }
            return false;
        } catch (error) {
            console.error('Error updating race:', error);
            return false;
        }
    }

    static deleteRace(id) {
        try {
            const races = this.getRaces();
            const filteredRaces = races.filter(race => race.id !== id);
            localStorage.setItem('races', JSON.stringify(filteredRaces));
            return true;
        } catch (error) {
            console.error('Error deleting race:', error);
            return false;
        }
    }

    // Horse management
    static addHorse(horseData) {
        try {
            const horses = this.getHorses();
            
            // Check for duplicate names
            if (horses.some(h => h.name.toLowerCase() === horseData.name.toLowerCase())) {
                throw new Error('A horse with this name already exists');
            }
            
            horseData.id = horseData.id || Date.now().toString();
            horseData.createdAt = new Date().toISOString();
            horses.push(horseData);
            localStorage.setItem('horses', JSON.stringify(horses));
            return true;
        } catch (error) {
            console.error('Error adding horse:', error);
            return false;
        }
    }

    static getHorses() {
        try {
            return JSON.parse(localStorage.getItem('horses') || '[]');
        } catch (error) {
            console.error('Error loading horses:', error);
            return [];
        }
    }

    static getHorseById(id) {
        const horses = this.getHorses();
        return horses.find(horse => horse.id === id);
    }

    // Prediction management
    static addPrediction(raceId, predictions) {
        try {
            const races = this.getRaces();
            const raceIndex = races.findIndex(race => race.id === raceId);
            if (raceIndex !== -1) {
                races[raceIndex].predictions = predictions;
                races[raceIndex].predictedAt = new Date().toISOString();
                localStorage.setItem('races', JSON.stringify(races));
                return true;
            }
            return false;
        } catch (error) {
            console.error('Error adding prediction:', error);
            return false;
        }
    }

    static makePrediction(raceId) {
        const race = this.getRaceById(raceId);
        if (!race || !race.horses || race.horses.length === 0) {
            console.error('Cannot make prediction: Race not found or no horses');
            return null;
        }

        // Generate predictions using the existing generatePrediction method
        const predictions = this.generatePrediction(race);
        
        if (!predictions || predictions.length === 0) {
            console.error('Failed to generate predictions');
            return null;
        }

        // Save the prediction using addPrediction method
        return this.addPrediction(raceId, predictions);
    }

    static getPredictions() {
        const races = this.getRaces();
        return races.filter(race => race.predictions && race.predictions.length > 0);
    }

    // Results management
    static addResult(raceId, results) {
        try {
            const races = this.getRaces();
            const raceIndex = races.findIndex(race => race.id === raceId);
            if (raceIndex !== -1) {
                races[raceIndex].results = results;
                races[raceIndex].completedAt = new Date().toISOString();
                localStorage.setItem('races', JSON.stringify(races));
                return true;
            }
            return false;
        } catch (error) {
            console.error('Error adding result:', error);
            return false;
        }
    }

    // Statistics
    static getStats() {
        const races = this.getRaces();
        const horses = this.getHorses();
        
        const completedRaces = races.filter(race => race.results);
        const racesWithPredictions = completedRaces.filter(race => race.predictions && race.predictions.length > 0);
        const correctPredictions = racesWithPredictions.filter(race => 
            race.predictions[0].horse === race.results.winner
        );
        
        const predictionAccuracy = racesWithPredictions.length > 0 ? 
            (correctPredictions.length / racesWithPredictions.length * 100) : 0;
        
        const totalPrizeMoney = races.reduce((sum, race) => sum + (race.prizeMoney || 0), 0);
        
        return {
            totalRaces: races.length,
            totalHorses: horses.length,
            completedRaces: completedRaces.length,
            predictionAccuracy: predictionAccuracy.toFixed(1),
            totalPrizeMoney: totalPrizeMoney,
            correctPredictions: correctPredictions.length,
            totalPredictions: racesWithPredictions.length
        };
    }

    // Recent activity
    static getRecentActivity() {
        const races = this.getRaces();
        const horses = this.getHorses();
        const activities = [];

        // Recent races
        const recentRaces = races
            .filter(race => race.createdAt)
            .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
            .slice(0, 3);

        recentRaces.forEach(race => {
            activities.push({
                description: `Race "${race.name}" added`,
                date: race.createdAt,
                type: 'race'
            });
        });

        // Recent horses
        const recentHorses = horses
            .filter(horse => horse.createdAt)
            .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
            .slice(0, 2);

        recentHorses.forEach(horse => {
            activities.push({
                description: `Horse "${horse.name}" added`,
                date: horse.createdAt,
                type: 'horse'
            });
        });

        // Recent predictions
        const racesWithPredictions = races
            .filter(race => race.predictions && race.predictedAt)
            .sort((a, b) => new Date(b.predictedAt) - new Date(a.predictedAt))
            .slice(0, 2);

        racesWithPredictions.forEach(race => {
            activities.push({
                description: `Prediction made for "${race.name}"`,
                date: race.predictedAt,
                type: 'prediction'
            });
        });

        return activities
            .sort((a, b) => new Date(b.date) - new Date(a.date))
            .slice(0, 5);
    }

    // Simple prediction algorithm
    static generatePrediction(race) {
        if (!race.horses || race.horses.length === 0) {
            return [];
        }

        // Calculate scores for each horse based on various factors
        const predictions = race.horses.map(horse => {
            let score = Math.random() * 100; // Base random score
            
            // Adjust based on odds (lower odds = higher chance)
            if (horse.odds) {
                const oddsValue = parseFloat(horse.odds);
                if (!isNaN(oddsValue)) {
                    score += (20 - Math.min(oddsValue, 20)) * 2; // Boost for lower odds
                }
            }
            
            // Adjust based on weight (lighter is generally better)
            if (horse.weight) {
                const weight = parseFloat(horse.weight);
                if (!isNaN(weight)) {
                    score += (60 - Math.min(weight, 60)) * 0.5; // Boost for lighter weight
                }
            }
            
            // Add some randomness for realism
            score += (Math.random() - 0.5) * 20;
            
            return {
                horse: horse.name,
                confidence: Math.min(Math.max(score / 100, 0.1), 0.95), // Normalize to 0.1-0.95
                factors: {
                    odds: horse.odds,
                    weight: horse.weight,
                    form: 'Unknown'
                }
            };
        });

        // Sort by confidence and return top 3
        return predictions
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, 3);
    }

    // API simulation methods
    static async fetchRacesFromAPI(provider, options = {}) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
        
        const { dateFrom, dateTo, maxRaces = 10 } = options;
        
        // Generate sample race data based on provider
        const sampleRaces = this.generateSampleRaces(provider, maxRaces, dateFrom, dateTo);
        
        return {
            success: true,
            races: sampleRaces,
            provider: provider,
            timestamp: new Date().toISOString()
        };
    }

    static generateSampleRaces(provider, count, dateFrom, dateTo) {
        const tracks = ['Churchill Downs', 'Belmont Park', 'Santa Anita', 'Keeneland', 'Saratoga', 'Del Mar'];
        const raceTypes = ['Maiden', 'Allowance', 'Stakes', 'Claiming', 'Handicap'];
        const distances = [1200, 1400, 1600, 1800, 2000, 2400];
        
        const races = [];
        const startDate = dateFrom ? new Date(dateFrom) : new Date();
        const endDate = dateTo ? new Date(dateTo) : new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);
        
        for (let i = 0; i < count; i++) {
            const raceDate = new Date(startDate.getTime() + Math.random() * (endDate.getTime() - startDate.getTime()));
            const track = tracks[Math.floor(Math.random() * tracks.length)];
            const raceType = raceTypes[Math.floor(Math.random() * raceTypes.length)];
            const distance = distances[Math.floor(Math.random() * distances.length)];
            
            const horses = this.generateSampleHorses(6 + Math.floor(Math.random() * 6)); // 6-12 horses
            
            races.push({
                id: `api_${Date.now()}_${i}`,
                name: `${raceType} Race ${i + 1}`,
                track: track,
                date: raceDate.toISOString(),
                distance: distance,
                raceNumber: i + 1,
                prizeMoney: 10000 + Math.random() * 90000,
                horses: horses,
                source: provider,
                createdAt: new Date().toISOString()
            });
        }
        
        return races;
    }

    static generateSampleHorses(count) {
        const horseNames = [
            'Thunder Strike', 'Lightning Bolt', 'Storm Chaser', 'Wind Runner', 'Fire Spirit',
            'Golden Arrow', 'Silver Bullet', 'Midnight Express', 'Dawn Breaker', 'Star Gazer',
            'Ocean Wave', 'Mountain Peak', 'Desert Storm', 'Forest Fire', 'Ice Crystal'
        ];
        
        const jockeys = [
            'J. Smith', 'M. Johnson', 'R. Williams', 'S. Brown', 'T. Davis',
            'A. Miller', 'C. Wilson', 'D. Moore', 'E. Taylor', 'F. Anderson'
        ];
        
        const horses = [];
        const usedNames = new Set();
        
        for (let i = 0; i < count; i++) {
            let name;
            do {
                name = horseNames[Math.floor(Math.random() * horseNames.length)];
            } while (usedNames.has(name));
            usedNames.add(name);
            
            horses.push({
                name: name,
                jockey: jockeys[Math.floor(Math.random() * jockeys.length)],
                weight: (52 + Math.random() * 8).toFixed(1), // 52-60 kg
                odds: (2 + Math.random() * 18).toFixed(1), // 2-20 odds
                number: i + 1
            });
        }
        
        return horses;
    }

    static importRacesFromAPI(apiRaces) {
        try {
            const existingRaces = this.getRaces();
            let importedCount = 0;
            
            apiRaces.forEach(apiRace => {
                // Check if race already exists (by name, track, and date)
                const exists = existingRaces.some(race => 
                    race.name === apiRace.name && 
                    race.track === apiRace.track && 
                    race.date.split('T')[0] === apiRace.date.split('T')[0]
                );
                
                if (!exists) {
                    existingRaces.push(apiRace);
                    importedCount++;
                }
            });
            
            localStorage.setItem('races', JSON.stringify(existingRaces));
            return importedCount;
        } catch (error) {
            console.error('Error importing races:', error);
            return 0;
        }
    }

    // Utility methods
    static formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    }

    static formatDate(dateString) {
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    }

    static formatTime(dateString) {
        return new Date(dateString).toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    // Data export/import
    static exportData() {
        const data = {
            races: this.getRaces(),
            horses: this.getHorses(),
            exportedAt: new Date().toISOString()
        };
        return JSON.stringify(data, null, 2);
    }

    static importData(jsonData) {
        try {
            const data = JSON.parse(jsonData);
            if (data.races) {
                localStorage.setItem('races', JSON.stringify(data.races));
            }
            if (data.horses) {
                localStorage.setItem('horses', JSON.stringify(data.horses));
            }
            return true;
        } catch (error) {
            console.error('Error importing data:', error);
            return false;
        }
    }

    // Clear all data
    static clearAllData() {
        localStorage.removeItem('races');
        localStorage.removeItem('horses');
        return true;
    }
}

// Make DataManager available globally
window.DataManager = DataManager;